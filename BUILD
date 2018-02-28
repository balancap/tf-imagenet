# Description:
#   Contains files for loading, training and evaluating TF-Slim-based models.

package(
    default_visibility = ["//visibility:public"],
)

sh_binary(
    name = "download_and_convert_imagenet",
    srcs = ["datasets/tfrecords/download_and_convert_imagenet.sh"],
    data = [
        "datasets/tfrecords/download_imagenet.sh",
        "datasets/tfrecords/imagenet_2012_validation_synset_labels.txt",
        "datasets/tfrecords/imagenet_lsvrc_2015_synsets.txt",
        "datasets/tfrecords/imagenet_metadata.txt",
        "datasets/tfrecords/preprocess_imagenet_validation_data.py",
        "datasets/tfrecords/process_bounding_boxes.py",
        ":build_imagenet_data",
    ],
)

py_binary(
    name = "build_imagenet_data",
    srcs = ["datasets/tfrecords/build_imagenet_data.py"],
    deps = [
        # "//numpy",
        # "//tensorflow",
    ],
)
