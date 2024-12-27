#!/bin/bash
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"
echo ${ScriptPath}

ModelPath="./model/bce-rerank/bce_rerank_bs1-32_1-512_linux_aarch64.om"
SentenceModel="./model/bce-rerank/sentencepiece.bpe.model"
echo ${ModelPath}

if [[ -n "$1" && -n "$2" ]]; then
     ModelPath=$1
     SentenceModel=$2
fi
 

# common_script_dir=${THIRDPART_PATH}/common
# . ${common_script_dir}/sample_common.sh
source ~/.bashrc;
source /usr/local/Ascend/ascend-toolkit/set_env.sh;


function main()
{
  echo "[INFO] The sample starts to run"

  running_command="./out/main"
  data_command=""
  
  ${running_command} -p ${ModelPath} -s ${SentenceModel}
  if [ $? -ne 0 ];then
    return 1
  fi
}
main
