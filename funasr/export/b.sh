for i in `seq 17 49`; do
    python blade_quant_opt.py z_models_int8_encoders/int8.encoders.$i.mask.pt encoders_inputs.pth opt
    mv blade.int8.pt z_models_int8_encoders/blade.encoders.$i.int8+fp16.mask.pt
done
python blade_quant_opt.py z_models_int8_encoders/int8.encoders0.mask.pt encoders0_inputs.pth opt
mv blade.int8.pt z_models_int8_encoders/blade.encoders0.int8+fp16.mask.pt
