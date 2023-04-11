#!/bin/sh

username="bmvarbanov"
dataset="20230403-d3_rot-css-surface_circ-level_p0.001"
from_scrath=true


if $from_scrath;
then
    host="delftblue"
    data_dir="/scratch/${username}"
else
    host="linux-bastion.tudelft.nl"
    data_dir="/tudelft.net/staff-umbrella/qrennd"
fi

source_dir="${data_dir}/output/${dataset}"

script_dir=$PWD
destination_dir="${script_dir}/data"

rsync -avzh --progress --stats --no-perms "${username}@${host}:${source_dir}" "${destination_dir}"