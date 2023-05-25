echo $1 | awk '{print $1 / 1000.0 / 1000.0 / 100.0 "ms"}'
echo $1 $2 | awk '{print $1 / ($2 / 100.0) / 1000.0 / 1000.0 / 100.0 "ms"}'

# nsys nvprof --profile-from-start off
