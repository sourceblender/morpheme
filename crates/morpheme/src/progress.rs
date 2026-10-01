//! Trainer progress reporting.
//!
//! With the `progressbar` feature (on by default) this draws `indicatif`
//! bars on stderr; they are hidden automatically when stderr is not a
//! terminal. Without the feature, or when a trainer's `show_progress` is
//! false, every method is a no-op.

#[cfg(feature = "progressbar")]
use indicatif::{ProgressBar, ProgressStyle};

/// A progress bar (known length) or spinner (unknown length).
pub(crate) struct Progress {
    #[cfg(feature = "progressbar")]
    bar: Option<ProgressBar>,
}

impl Progress {
    /// A bar of `len` steps, or a spinner if `len` is `None`.
    pub(crate) fn new(enabled: bool, message: &str, len: Option<u64>) -> Self {
        #[cfg(feature = "progressbar")]
        {
            if !enabled {
                return Self { bar: None };
            }
            let bar = match len {
                Some(n) => {
                    let bar = ProgressBar::new(n);
                    bar.set_style(
                        ProgressStyle::with_template(
                            "[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:>9}/{len:<9}",
                        )
                        .expect("valid progress template"),
                    );
                    bar
                }
                None => {
                    let bar = ProgressBar::new_spinner();
                    bar.set_style(
                        ProgressStyle::with_template(
                            "[{elapsed_precise}] {msg:<30!} {spinner} {pos}",
                        )
                        .expect("valid progress template"),
                    );
                    bar.enable_steady_tick(std::time::Duration::from_millis(120));
                    bar
                }
            };
            bar.set_message(message.to_owned());
            Self { bar: Some(bar) }
        }
        #[cfg(not(feature = "progressbar"))]
        {
            let _ = (enabled, message, len);
            Self {}
        }
    }

    /// Advance by `n` steps.
    pub(crate) fn inc(&self, n: u64) {
        #[cfg(feature = "progressbar")]
        if let Some(bar) = &self.bar {
            bar.inc(n);
        }
        #[cfg(not(feature = "progressbar"))]
        let _ = n;
    }

    /// Jump to step `pos`.
    pub(crate) fn set_position(&self, pos: u64) {
        #[cfg(feature = "progressbar")]
        if let Some(bar) = &self.bar {
            bar.set_position(pos);
        }
        #[cfg(not(feature = "progressbar"))]
        let _ = pos;
    }

    /// Replace the message.
    pub(crate) fn set_message(&self, message: impl Into<String>) {
        #[cfg(feature = "progressbar")]
        if let Some(bar) = &self.bar {
            bar.set_message(message.into());
        }
        #[cfg(not(feature = "progressbar"))]
        let _ = message.into();
    }

    /// Mark as done, leaving the final state on its own line.
    pub(crate) fn finish(&self) {
        #[cfg(feature = "progressbar")]
        if let Some(bar) = &self.bar {
            bar.finish();
            // indicatif leaves the cursor at the end of the finished bar;
            // move to a fresh line so the next bar (or other output)
            // doesn't overwrite or append to it.
            if !bar.is_hidden() {
                eprintln!();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_and_enabled_progress_never_panic() {
        for enabled in [false, true] {
            for len in [None, Some(10)] {
                let p = Progress::new(enabled, "test", len);
                p.inc(3);
                p.set_position(5);
                p.set_message("still testing");
                p.finish();
            }
        }
    }
}
