mode=${1}
dataset_name=${2}

source_dir=/home/gregor/Documents/medicaldetectiontoolkit

exps_dir=/home/gregor/networkdrives/E132-Cluster-Projects/${dataset_name}/experiments_float_data
exps_dirs=$(ls -d ${exps_dir}/*)
for dir in ${exps_dirs}; do
	echo "starting ${mode} in ${dir}"
	(python ${source_dir}/exec.py --use_stored_settings --mode ${mode} --dataset_name ${dataset_name} --exp_dir ${dir}) || (echo "FAILED!")
done
