echo "building training_Dataset-1"
python lifting/pcode_lifter.py \
    --cfg_summary ./dbs/Dataset-1/cfg_summary/training \
    --output_dir ./dbs/Dataset-1/features/training/pcode_raw_Dataset-1_training \
    --graph_type ALL \
    --verbose 1 \
    --nproc 32

echo "building validation_Dataset-1"
python lifting/pcode_lifter.py \
    --cfg_summary ./dbs/Dataset-1/cfg_summary/validation \
    --output_dir ./dbs/Dataset-1/features/validation/pcode_raw_Dataset-1_validation \
    --graph_type ALL \
    --verbose 1 \
    --nproc 32

echo "building testing_Dataset-1"
python lifting/pcode_lifter.py \
    --cfg_summary ./dbs/Dataset-1/cfg_summary/testing \
    --output_dir ./dbs/Dataset-1/features/testing/pcode_raw_Dataset-1_testing \
    --graph_type ALL \
    --verbose 1 \
    --nproc 32
