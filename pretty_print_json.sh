#!/usr/bin/env bash

if [[ -z "$1" ]]; then
  echo "usage: ${0} [path_to_json_file]"
  exit
fi

cat $1 | python -m json.tool --sort-keys
