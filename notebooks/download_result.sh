#!/bin/sh

username="bmvarbanov"
dataset="20230214-d3_rot-surf_circ-level_p0.001"
run="20230222-161627_units64_p0.001_eval-dropout0.2_l2null_lr0.001_extended-val_aux-weight0.8"
from_scrath=true


if $from_scrath;
then
    host="login.delftblue.tudelft.nl"
    data_dir="/scratch/${username}"
else
    host="linux-bastion.tudelft.nl"
    data_dir="/tudelft.net/staff-umbrella/qrennd"
fi

source_dir="${data_dir}/output/${dataset}/${run}"

script_dir=$PWD
destination_dir="${script_dir}/data/${dataset}"

rsync -avzh --progress --stats --no-perms "${username}@${host}:${source_dir}" "${destination_dir}"