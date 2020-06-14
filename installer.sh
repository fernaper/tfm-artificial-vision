echo "Generating folders structure..."
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

mkdir -p logs/scalars
mkdir -p logs/server
mkdir -p checkpoints/alexnet
mkdir -p checkpoints/resnet
mkdir -p data
mkdir -p videos

cp tfm_core/tfm_core/config.default.py tfm_core/tfm_core/config.py


echo "Download compress files from Google Drive... (dataset, neural network models, etc.)"

if [ -d "tfm-extra/" ]; then
    echo "Skipped: Already downloaded."
else
    # https://drive.google.com/file/d/16x92kjHqq1XojZydG9v9lktPQc4JdOQK/view?usp=sharing
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16x92kjHqq1XojZydG9v9lktPQc4JdOQK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16x92kjHqq1XojZydG9v9lktPQc4JdOQK" -O tfm-extra.tar.gz && rm -rf /tmp/cookies.txt
    echo "Extract compress files"
    tar -xf tfm-extra.tar.gz
    echo "Remove .tar.gz"
    rm tfm-extra.tar.gz
fi

# Link auxiliar data

ln -s $(pwd)/tfm-extra/mio-tcd-final data/mio-tcd-final
ln -s $(pwd)/tfm-extra/20181006_130650.mp4 videos/20181006_130650.mp4
ln -s $(pwd)/tfm-extra/models models


# Install library
echo "Installing custom code as a python library (tfm_core)"
pip3 install -e $(pwd)/tfm_core/

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
echo "\tpython3 detect_yolo.py -S 0.5 -v ../../videos/20181006_130650.mp4"
echo "\nLaunch tensorboard (to check how is going the training and RNN models)"
echo "\t./tensorboard.sh"
echo "\nTest the complete system with Alexnet:"
echo "\n\t./tensorserver.sh alexnet"
echo "\tcd tfm_core/tfm_core/"
echo "\tpython3 detect.py -S 0.5 -v ../../videos/20181006_130650.mp4 -d mio-tcd-finals -m alexnet -i 227"
echo "\nTest the complete system with Resnet50:"
echo "\n\t./tensorserver.sh resnet"
echo "\tpython3 detect.py -S 0.5 -v ../../videos/20181006_130650.mp4 -d mio-tcd-finals -m resnet -i 64"
echo "\nTrain Alexnet (check progress on tensorboard):"
echo "\tpython3 train.py -m alexnet -f mio-tcd-finals -e 30 -b 16"
echo "\nTrain Resnet (check progress on tensorboard):"
echo "\tpython3 train.py -m resnet -f mio-tcd-finals -e 30 -b 32\n"
echo "\tpython3 train.py -m resnet-18 -f mio-tcd-finals -e 30 -b 32\n"
echo "\tpython3 train.py -m resnet-34 -f mio-tcd-finals -e 30 -b 32\n"
echo "\tpython3 train.py -m resnet-101 -f mio-tcd-finals -e 30 -b 16\n"