# TensorFlow Object Detection Starter
TensorFlow Object Detection Starter is a sample project also a guide of how you can train your own data for object detection.

Now we're working on the easiest one, which is called "Image Retraining".

### Preparation
- Install [Python](https://www.python.org) 3.6.x (At this moment the Tensorflow does not support Python 3.7.x)
- Install [TensorFlow](https://github.com/tensorflow/tensorflow) & [TensorFlow Hub](https://www.tensorflow.org/hub/)
```sh
pip install tensorflow
pip install tensorflow_hub
```

Note: if you have seen this kind of error: **`could not find a version that satisfies the requirement tensorflow`**, please Use Python 3.6.x.

- Prepare the images to be trained & verified (tested), please check them [here](samples/) for example, tag the pictures by naming the folders

### Retraining

There is an official tutorial of [How to Retrain an Image Classifier for New Categories](https://www.tensorflow.org/hub/tutorials/image_retraining)

```sh
python retrain.py --image_dir=samples/retrain/ \
--saved_model_dir=result/saved_model/ \
--output_graph=result/foo.pb \
--output_labels=result/foo.txt \
-—bottleneck_dir=result/bottleneck/ \
--summaries_dir=result/retrain_logs/ \
-—intermediate_output_graphs_dir=result/intermediate_graph/ \
--tfhub_module=https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/1 \
--how_many_training_steps=4000
```

After running through the training process, you'll get the trained `model` (foo.pb), `label` (foo.txt) and `saved_model` files from the `result` folder.


### Verify the trained model
Verify the model with `label_image.py`, which comes from the official tensorflow repository, and we only changed the default settings so we don't have to type too many parameters. Check out the [original folder here for label_image.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image)

Execute the `label_image.py`:
```sh
python label_image.py
```
And you should see the output something like:
```properties
r2d2 0.99754924
bb8 0.0024507595
```

### Convert trained model to TFLite format

- There is an official link of Tensorflow TFLite: [Introduction to TensorFlow Lite](https://www.tensorflow.org/lite/overview)

For the trained data, we can easily use `tflite_convert` command like this:
```sh
tflite_convert --output_file=result/foo.tflite --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --input_array=Placeholder --output_array=final_result --inference_type=FLOAT --input_data_type=FLOAT --graph_def_file=result/foo.pb
```
