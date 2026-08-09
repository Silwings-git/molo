//! Shared state: a type-safe heterogeneous container.
//!
//! Tools receive it at **call time** via the `state` parameter of
//! [`Tool::call`](crate::Tool::call): the state is owned by the caller and
//! passed in on every call; tools themselves hold no global state. Agents
//! mount it via [`with_state`](crate::agent::ReActAgent::with_state), the
//! application reads and writes across runs, and multiple tools / agents
//! can share the same instance.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};

/// Shared state: a heterogeneous container accessed by type.
///
/// Type is the key: a second [`insert`](SharedState::insert) of the same
/// type overwrites; retrieval takes a type parameter (compile-time type
/// safety). Unlike `Box<dyn Any>`'s "store anything", this container holds
/// **multiple values side by side, distinguished by type**.
///
/// An internal `RwLock` (read-heavy) provides cross-thread safety;
/// [`Clone`] is cheap (Arc), so tools and agents — several agents — share
/// the same instance.
///
/// # Example
///
/// ```rust
/// use molo::SharedState;
///
/// #[derive(Debug, Clone, PartialEq)]
/// struct Session { user: String }
///
/// let state = SharedState::new();
/// state.insert(Session { user: "alice".into() });
/// state.with::<Session>(|s| assert_eq!(s.user, "alice"));
/// state.with_mut::<Session>(|s| s.user = "bob".into());
/// assert_eq!(state.get::<Session>(), Some(Session { user: "bob".into() }));
/// ```
#[derive(Clone, Default)]
pub struct SharedState {
    inner: Arc<RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>,
}

impl fmt::Debug for SharedState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Inner values are opaque `dyn Any`; printing the entry count is
        // enough (the caller knows the content types).
        let entries = self.inner.read().unwrap_or_else(|e| e.into_inner()).len();
        f.debug_struct("SharedState")
            .field("entries", &entries)
            .finish()
    }
}

impl SharedState {
    /// An empty shared state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a custom value; storing the same type again overwrites.
    pub fn insert<T: Any + Send + Sync>(&self, value: T) {
        // A panicking user closure (see with_mut) poisons the lock: recover
        // from poisoning so the container does not fail permanently, and
        // shared state stays usable after tool panics are caught.
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(TypeId::of::<T>(), Box::new(value));
    }

    /// Retrieve (clone) a custom value; returns `None` when the type is
    /// absent (the type must implement `Clone`; a clone is returned and the
    /// original value in the container is unchanged).
    pub fn get<T: Any + Clone>(&self) -> Option<T> {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&TypeId::of::<T>())
            .and_then(|v| v.downcast_ref::<T>())
            .cloned()
    }

    /// Read an existing value under the lock; the closure does not run when
    /// the type is absent.
    ///
    /// # Panics
    ///
    /// A panic inside the closure propagates outward; the lock gets
    /// poisoned as a result, but the container recovers from poisoning and
    /// later access works as usual (the closure returns nothing; for an
    /// in-place read use [`get`](SharedState::get)).
    ///
    /// # Deadlock red line
    ///
    /// The underlying `RwLock` is **not reentrant**, and the closure runs
    /// while the lock is held: the closure **must not** access the same
    /// `SharedState` again (e.g. `with(|a| ... state.get::<B>() ...)`), or
    /// it hangs with no error. To access several values at once, call
    /// `get` sequentially / combine them into one struct, or move reads
    /// and writes outside the closure.
    pub fn with<T: Any>(&self, f: impl FnOnce(&T)) {
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());
        if let Some(value) = guard
            .get(&TypeId::of::<T>())
            .and_then(|v| v.downcast_ref::<T>())
        {
            f(value);
        }
    }

    /// Update an existing value in place under the lock; the closure does
    /// not run when the type is absent.
    ///
    /// # Panics
    ///
    /// A panic inside the closure propagates outward; the lock gets
    /// poisoned as a result, but the container recovers from poisoning and
    /// later access works as usual.
    ///
    /// # Deadlock red line
    ///
    /// Same as [`with`](SharedState::with): the underlying lock is not
    /// reentrant, and the closure must not access the same `SharedState`
    /// again.
    pub fn with_mut<T: Any>(&self, f: impl FnOnce(&mut T)) {
        let mut guard = self.inner.write().unwrap_or_else(|e| e.into_inner());
        if let Some(value) = guard
            .get_mut(&TypeId::of::<T>())
            .and_then(|v| v.downcast_mut::<T>())
        {
            f(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Session {
        user: String,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Counter(usize);

    #[test]
    fn insert_and_get_by_type() {
        let state = SharedState::new();
        state.insert(Session {
            user: "alice".into(),
        });
        state.insert(Counter(3));

        assert_eq!(
            state.get::<Session>(),
            Some(Session {
                user: "alice".into()
            })
        );
        assert_eq!(state.get::<Counter>(), Some(Counter(3)));
    }

    /// Multiple values coexist, distinguished by type; a second insert of
    /// the same type overwrites.
    #[test]
    fn same_type_insert_overwrites_others_untouched() {
        let state = SharedState::new();
        state.insert(Session {
            user: "alice".into(),
        });
        state.insert(Counter(1));
        state.insert(Counter(2)); // same type overwrites

        assert_eq!(state.get::<Counter>(), Some(Counter(2)));
        assert_eq!(
            state.get::<Session>(),
            Some(Session {
                user: "alice".into()
            })
        );
    }

    #[test]
    fn get_missing_type_returns_none() {
        let state = SharedState::new();
        assert_eq!(state.get::<Session>(), None);
    }

    #[test]
    fn with_and_with_mut_lock_inner() {
        let state = SharedState::new();
        state.insert(Counter(1));

        // Missing type: the closure does not run.
        let mut called = false;
        state.with::<Session>(|_| called = true);
        assert!(!called);

        state.with::<Counter>(|c| {
            assert_eq!(c.0, 1);
        });
        state.with_mut::<Counter>(|c| c.0 += 1);
        assert_eq!(state.get::<Counter>(), Some(Counter(2)));
    }

    /// Clone sharing: two holders share one instance; writes from one are
    /// visible to the other.
    #[test]
    fn clone_shares_same_instance() {
        let state = SharedState::new();
        let tool_side = state.clone();

        state.insert(Counter(5));
        assert_eq!(tool_side.get::<Counter>(), Some(Counter(5)));
        tool_side.with_mut::<Counter>(|c| c.0 += 1);
        assert_eq!(state.get::<Counter>(), Some(Counter(6)));
    }

    /// After a user closure panics and poisons the lock, the container
    /// recovers from poisoning and later access works normally.
    #[test]
    fn poisoned_lock_recovers_and_stays_usable() {
        let state = SharedState::new();
        state.insert(Counter(1));

        // Closure panics: the lock is poisoned.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.with_mut::<Counter>(|_| panic!("user closure panicked"));
        }));
        assert!(panicked.is_err());

        // Recovered from poisoning: later reads and writes work as usual.
        state.with_mut::<Counter>(|c| c.0 += 1);
        assert_eq!(state.get::<Counter>(), Some(Counter(2)));
    }
}
