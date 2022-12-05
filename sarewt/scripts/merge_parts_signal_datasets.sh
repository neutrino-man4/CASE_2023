#!/bin/bash

base_dir_in="/eos/project/d/dshep/TOPCLASS/DijetAnomaly/"
base_dir_out="/eos/user/k/kiwoznia/data/VAE_data/events/"

for mass_val in 1.5 2.5 3.5 4.5
do
    # gravitons
    for resonance in NARROW BROAD
    do
        sample_name="RSGraviton_WW_${resonance}_13TeV_PU40_${mass_val}TeV_NEW"
        dir_in="${base_dir_in}${sample_name}"
        dir_out="${base_dir_out}${sample_name}_parts/"
        echo "${dir_in} ${dir_out}"
        mkdir -p "${dir_out}" 
        python3 -u event_concatenate_serialization.py -in "${dir_in}" -out "${dir_out}${sample_name}_concat.h5" -mb 1500
    done

    # A -> ZZZ
    sample_name="AtoHZ_to_ZZZ_13TeV_PU40_${mass_val}TeV_NEW"
    dir_in="${base_dir_in}${sample_name}"
    dir_out="${base_dir_out}${sample_name}_parts/"
    echo "${dir_in} ${dir_out}"
    mkdir -p "${dir_out}" 
    python3 -u event_concatenate_serialization.py -in "${dir_in}" -out "${dir_out}${sample_name}_concat.h5" -mb 1500

done

