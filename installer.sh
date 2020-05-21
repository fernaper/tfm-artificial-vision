echo "Generating folders structure..."

mkdir logs
mkdir logs/scalars
mkdir logs/server
mkdir models
mkdir models/alexnet
mkdir models/resnet
mkdir checkpoints
mkdir checkpoints/alexnet
mkdir checkpoints/resnet

cp tfm_core/tfm_core/config.default.py tfm_core/tfm_core/config.py

# Download own yolo-coco


# Download own data/mio-tcd_dataset


# Download videos


echo "Easy steps to test the system. I recommend running each command first with the -h parameter to see check all alternatives.\n"
echo "\nTest utilities video controller (foundation for the rest):"
echo "\tcd tfm_core/tfm_core/"
echo "\tpython3 utilities.py -v ../../videos/20181006_130650.mp4"
echo "\nTest OpenCV movement detector:"
echo "\tcd tfm_core/tfm_core/"
echo "\tpython3 movement_detection.py -v ../../videos/20181006_130650.mp4"
echo "\nTest Optical flow:"
echo "\tcd tfm_core/tfm_core/"
echo "\tpython3 optical_flow.py -v ../../videos/cars1.mp4 -a dense -c"
echo "\nTest YOLO: "
echo "\tcd tfm_core/tfm_core/"
echo "\tpython3 dense_clasifier.py -S 0.5 -v ../../videos/20181006_130650.mp4 --yolo yolo-coco"
echo "\nLaunch tensorboard (to check how is going the training and RNN models)"
echo "\t./tensorboard.sh"
echo "\nTest the complete system with Alexnet:"
echo "\n\t./tensorserver.sh alexnet"
echo "\tcd tfm_core/tfm_core/"
echo "\tpython3 dense_clasifier_tf.py -S 0.5 -v ../../videos/20181006_130650.mp4 -d medium_mio-tcd_dataset_9class -m alexnet -i 227"
echo "\nTest the complete system with Resnet50:"
echo "\n\t./tensorserver.sh resnet"
echo "\tpython3 dense_clasifier_tf.py -S 0.5 -v ../../videos/20181006_130650.mp4 -d medium_mio-tcd_dataset_9class -m resnet -i 64"
echo "\nTrain Alexnet (check progress on tensorboard):"
echo "\tpython3 train.py -m alexnet -f medium_mio-tcd_dataset_9class -e 30 -b 16"
echo "\nTrain Resnet (check progress on tensorboard):"
echo "\tpython3 train.py -m resnet -f medium_mio-tcd_dataset_9class -e 30 -b 16\n"