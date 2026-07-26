"""Compatibility entry point named by the DISC reproduction protocol."""

try:
    from eval_relm import main
except ImportError:  # Supports ``python -m tools.eval_relm_disc``.
    from tools.eval_relm import main


if __name__ == "__main__":
    main()
