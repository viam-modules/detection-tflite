#!/bin/sh

cleanup() {
    rm -f labels.txt detection.tflite
}

python3 setup.py sdist --formats=gztar
pip install dist/model-0.1.tar.gz

python3 -m model.training --dataset_file=dataset.jsonl --model_output_directory=. --num_epochs=2
pip install pytest && pytest tests/test_file_exists.py

cleanup()

python3 -m model.training --dataset_file=dataset.jsonl --model_output_directory=. --num_epochs=2 --labels=green_square/blue_star
pytest tests/

cleanup()
