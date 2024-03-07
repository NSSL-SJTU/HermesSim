#!/bin/sh

DATASET_NAME=${DATASET_NAME:-'Dataset-1'}
FEAT=${FEAT:-'pcode_raw'}
DBDIR=${DBDIR:-'dbs'}
OUTDIR=${OUTDIR:-'inputs/pcode'}

echo "Processing ${DATASET_NAME}_training"
python preprocess/preprocessing_pcode.py \
    --training \
    --freq-mode -f pkl -s ${DATASET_NAME}_training \
    -i $DBDIR/$DATASET_NAME/features/training/"$FEAT"_${DATASET_NAME}_training \
    -o $OUTDIR

echo "Processing ${DATASET_NAME}_validation"
python preprocess/preprocessing_pcode.py \
    --freq-mode -f pkl -s ${DATASET_NAME}_validation \
    -i $DBDIR/$DATASET_NAME/features/validation/"$FEAT"_${DATASET_NAME}_validation \
    -o $OUTDIR

echo "Processing ${DATASET_NAME}_testing"
python preprocess/preprocessing_pcode.py \
    --freq-mode -f pkl -s ${DATASET_NAME}_testing \
    -i $DBDIR/$DATASET_NAME/features/testing/"$FEAT"_${DATASET_NAME}_testing \
    -o $OUTDIR

