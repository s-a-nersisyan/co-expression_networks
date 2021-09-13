config_path=$1
echo $config_path

python3 test.py $config_path
echo "test"
python3 hyperise.py $config_path -o
echo "hyperise"
python3 binomise.py $config_path -o
echo "binomise"
# python3 cluster.py $config_path -o
# echo "cluster"
