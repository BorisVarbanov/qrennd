#!/bin/sh

username="bmvarbanov"
dataset="20230306-d3_rot-surf_biased-noise"
run="20230326-161328_google_simulated_dr0-05_dim128_continue2"
from_scrath=false


if $from_scrath;
then
    host="delftblue"
    data_dir="/scratch/${username}"
else
    host="linux-bastion.tudelft.nl"
    data_dir="/tudelft.net/staff-umbrella/qrennd"
fi

source_dir="${data_dir}/output/${dataset}/${run}"

script_dir=$PWD
destination_dir="${script_dir}/data/${dataset}"

rsync -avzh --progress --stats --no-perms "${username}@${host}:${source_dir}" "${destination_dir}"