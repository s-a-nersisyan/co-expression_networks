config_path=$1

python3 exhaustive_test.py $config_path
echo "exhaustive_test"
python3 exhaustive_hyperise.py $config_path
echo "exhaustive_hyperise"
python3 exhaustive_binomise.py $config_path
echo "exhaustive_binomise"
python3 exhaustive_cluster.py $config_path
echo "exhaustive_cluster"
