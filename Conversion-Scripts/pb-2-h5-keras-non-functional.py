
import argparse
import os
import subprocess
import sys

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])
    import tensorflow as tf



def convert(pb_file_path, h5_file_path):
    model = tf.saved_model.load(pb_file_path)
    infer = model.signatures["serving_default"]
    keras_model = tf.keras.Model(inputs=infer.inputs[0], outputs=infer.outputs[0])
    keras_model.save(h5_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a TensorFlow .pb model to a .h5 model.')
    parser.add_argument('pb_file', type=str, help='Path to the input .pb')
    parser.add_argument('-o', '--output', type=str, help='Path to the output .h5 (default: current dir)')
    args = parser.parse_args()

    if args.output:
        h5_file_path = args.output
    else:
        # Default to the same directory as the script with the same name as the .pb file
        h5_file_path = os.path.splitext(args.pb_file)[0] + '.h5'

    # Convert the model
    convert(args.pb_file, h5_file_path)
    print(f"Conversion success. from {args.pb_file} -> {h5_file_path}")