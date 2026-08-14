//! A command-line message channel: prints messages to the terminal and reads replies from stdin.

use super::{ChannelError, MessageChannel};
use std::fmt;
use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
};
use tokio::sync::Mutex;

/// Upper bound for a single input line (1 MiB): unbounded appends would exhaust memory when piped
/// input has no newline for a long time.
const MAX_LINE: usize = 1 << 20;

/// A command-line message channel: prints messages to the terminal and reads one line from stdin as the reply.
///
/// Binds to standard input / standard output by default; alternatively, inject custom read/write sources
/// with [`CliMessageChannel::with_io`] (files, network, in-memory buffers — common in tests).
///
/// Messages travel line by line: each message gets its own line on output (a newline is appended
/// automatically); `ask` reads one line from the input, trims leading and trailing whitespace,
/// and returns it as the reply.
///
/// All operations are serialized through an internal mutex: `ask` presents only one request at a time,
/// and other `ask` / `notify` calls queue while it waits for the reply; input is read asynchronously,
/// yielding while waiting, so other tasks are not blocked on a single-threaded runtime.
///
/// # Example
///
/// ```rust
/// # extern crate molo_agent as molo;
/// use molo::{CliMessageChannel, MessageChannel};
/// use tokio::io::{AsyncWriteExt, BufReader};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Inject in-memory read/write sources instead of touching a real terminal.
/// let (mut seed, rx) = tokio::io::duplex(64);
/// seed.write_all("yes\n".as_bytes()).await?;
/// let channel = CliMessageChannel::with_io(
///     Box::new(BufReader::new(rx)),
///     Box::new(tokio::io::sink()),
/// );
///
/// let answer = channel.ask("continue? [y/n]").await?;
/// assert_eq!(answer, "yes");
/// # Ok(())
/// # }
/// ```
///
/// # Notes
///
/// - when input reaches end-of-file (Ctrl-D / EOF), `ask` returns
///   [`ChannelError::Closed`](crate::ChannelError::Closed);
/// - replies are read line by line and trimmed: an empty line still counts as a valid reply (an empty string).
pub struct CliMessageChannel {
    io: Mutex<CliIo>,
}

impl fmt::Debug for CliMessageChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The read/write sources are trait objects and can't implement Debug; print the bound shapes
        // so logs can tell how the channel was constructed.
        f.debug_struct("CliMessageChannel")
            .field("io", &"Box<dyn AsyncBufRead> + Box<dyn AsyncWrite>")
            .finish()
    }
}

struct CliIo {
    reader: Box<dyn AsyncBufRead + Unpin + Send>,
    writer: Box<dyn AsyncWrite + Unpin + Send>,
}

impl CliMessageChannel {
    /// Binds to standard input / standard output.
    pub fn new() -> Self {
        Self::with_io(
            Box::new(BufReader::new(tokio::io::stdin())),
            Box::new(tokio::io::stdout()),
        )
    }

    /// Binds custom read/write sources.
    ///
    /// `reader` supplies the reply input (read line by line), and `writer` receives the output messages
    /// (written line by line and flushed). Tests can inject an in-memory buffer, or the channel can be
    /// attached to a file or the network.
    pub fn with_io(
        reader: Box<dyn AsyncBufRead + Unpin + Send>,
        writer: Box<dyn AsyncWrite + Unpin + Send>,
    ) -> Self {
        Self {
            io: Mutex::new(CliIo { reader, writer }),
        }
    }
}

impl Default for CliMessageChannel {
    fn default() -> Self {
        Self::new()
    }
}

/// Writes one line of message to the writer and flushes it.
async fn write_line(
    writer: &mut (dyn AsyncWrite + Unpin + Send),
    message: &str,
) -> Result<(), ChannelError> {
    writer
        .write_all(message.as_bytes())
        .await
        .map_err(ChannelError::from)?;
    writer.write_all(b"\n").await.map_err(ChannelError::from)?;
    writer.flush().await.map_err(ChannelError::from)
}

#[async_trait::async_trait]
impl MessageChannel for CliMessageChannel {
    async fn ask(&self, message: &str) -> Result<String, ChannelError> {
        let mut io = self.io.lock().await;
        write_line(io.writer.as_mut(), message).await?;
        // Bounded read: lines longer than MAX_LINE bytes are truncated and rejected (bounded memory when
        // piped input has no newline; the error includes the length for diagnosis).
        let mut line = String::new();
        let read = io
            .reader
            .as_mut()
            .take((MAX_LINE + 1) as u64)
            .read_line(&mut line)
            .await
            .map_err(ChannelError::from)?;
        if line.len() > MAX_LINE {
            return Err(ChannelError::Io(format!(
                "input line exceeds {MAX_LINE} bytes"
            )));
        }
        if read == 0 {
            return Err(ChannelError::Closed);
        }
        Ok(line.trim().to_string())
    }

    async fn notify(&self, message: &str) -> Result<(), ChannelError> {
        let mut io = self.io.lock().await;
        write_line(io.writer.as_mut(), message).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex as StdMutex};
    use std::task::{Context, Poll};

    /// Shared output buffer: collects written content across tasks in concurrency tests.
    #[derive(Clone, Default)]
    struct SharedBuf(Arc<StdMutex<Vec<u8>>>);

    impl AsyncWrite for SharedBuf {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            self.0
                .lock()
                .expect("internal lock poisoned")
                .extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    /// Builds a channel from the injected reader / writer; seed is the pre-written input content.
    async fn channel(seed: &str) -> (CliMessageChannel, SharedBuf) {
        let buf = SharedBuf(Arc::new(StdMutex::new(Vec::new())));
        let (mut seed_tx, rx) = tokio::io::duplex(64);
        seed_tx.write_all(seed.as_bytes()).await.unwrap();
        let channel =
            CliMessageChannel::with_io(Box::new(BufReader::new(rx)), Box::new(buf.clone()));
        (channel, buf)
    }

    #[tokio::test]
    async fn ask_reads_line_and_trims() {
        let (channel, buf) = channel("  reply text  \n").await;
        assert_eq!(channel.ask("question").await.unwrap(), "reply text");
        let out = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
        assert_eq!(out, "question\n");
    }

    #[tokio::test]
    async fn ask_closed_input_returns_closed() {
        let channel = CliMessageChannel::with_io(
            Box::new(BufReader::new(tokio::io::empty())),
            Box::new(tokio::io::sink()),
        );
        assert!(matches!(
            channel.ask("question").await,
            Err(ChannelError::Closed)
        ));
    }

    #[tokio::test]
    async fn notify_writes_message() {
        let (channel, buf) = channel("").await;
        channel.notify("notice").await.unwrap();
        let out = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
        assert_eq!(out, "notice\n");
    }

    #[tokio::test]
    async fn concurrent_asks_serialized() {
        let (channel, buf) = channel("r1\nr2\n").await;
        // Two asks run concurrently; the channel presents them serially, so replies line up one-to-one
        // without interleaving.
        let (a, b) = tokio::join!(channel.ask("m1"), channel.ask("m2"));
        let mut replies = vec![a.unwrap(), b.unwrap()];
        replies.sort();
        assert_eq!(replies, vec!["r1", "r2"]);
        // The output is also non-interleaved: two lines, each a complete message.
        let out = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines.contains(&"m1") && lines.contains(&"m2"));
    }
}
