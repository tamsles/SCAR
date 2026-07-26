"""Quick CPU-friendly synthetic demonstration."""

from train import main


if __name__ == "__main__":
    main(
        [
            "--config",
            "configs/ratio_synthetic.json",
            "--output-dir",
            "outputs/demo",
            "--device",
            "auto",
            "--epochs",
            "5",
        ]
    )

