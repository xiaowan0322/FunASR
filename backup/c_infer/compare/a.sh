LD_PRE=$LD_LIBRARY_PATH
LD="LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/python3.8/dist-packages/torch/lib:/usr/local/lib/python3.8/dist-packages/torch_blade:/usr/local/TensorRT/lib/:/usr/local/cuda/lib64/:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export LD_LIBRARY_PATH=$LD


nvidia-cuda-mps-control -d

if [ "$1" = "python" ]; then
    cmd="python torch_infer.py"
else
    cmd="./torch_app"
fi

num=1435
for i in `seq 0 4`; do
    start_i=$(( $i * $num))
    end_i=$(( $i * $num + $num))
    nohup $cmd model.fixfp16.blade.fp16.pt $start_i $end_i > log.$i 2>&1 &
done

wait
cat log.* | grep "t1"
cat log.* | grep "t1" | awk 'BEGIN{a=0} {if(a<$2){a=$2}} END{print "total_time: " a ", speed: " 36108918 / a}'

ps -ef | grep "nvidia-cuda-mps-" | grep -v "grep" | awk '{print $2}' | xargs kill -9

export LD_LIBRARY_PATH=$LD_PRE
