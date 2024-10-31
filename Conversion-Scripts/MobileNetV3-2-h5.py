import argparse
import os
import subprocess
import sys

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])
    import tensorflow as tf

def convert(h5_file_path):
    from tensorflow.keras.applications import MobileNetV3Small

    # Load the MobileNetV3 model
    model = MobileNetV3Small(weights='imagenet')
    #infer = model.signatures["serving_default"]
    #model = tf.keras.Model(inputs=infer.inputs[0], outputs=infer.outputs[0])

    # Save the model as a .h5 file
    model.save(h5_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert MobileNetV3 to .h5')
    parser.add_argument('-o', '--output', type=str, help='Path to the output .h5 (default: save in script directory)')
    args = parser.parse_args()

    # Determine the output path
    if args.output:
        h5_file_path = args.output
    else:
        # Save to the same directory as the script
        script_directory = os.path.dirname(os.path.abspath(__file__))
        h5_file_path = os.path.join(script_directory, 'mobilenet_v3_model.h5')  # Default output name

    # Convert the model
    convert(h5_file_path)
    print(f"Conversion success! Saved to: {h5_file_path}")
