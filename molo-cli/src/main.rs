fn main() {
    if let Err(error) = molo_cli::run_blocking(std::env::args().skip(1)) {
        eprintln!("molo: {error}");
        std::process::exit(error.exit_code());
    }
}
