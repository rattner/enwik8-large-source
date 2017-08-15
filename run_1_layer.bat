REM run 1 layer over the whole data until convergence
python enwik8_run.py --name=1_layer_ --num_layers=1 --sweeps=0 --units_num=1500 --time_steps=50 --drop_output_init=0.08 --drop_output_step=0.1 --drop_state_init=0.08 --drop_state_step=0.1 --drop_emb=0.2 --batch_size=100 --train_size=95000000 --valid_size=5000000

REM run the layer over a small coherent part of the data for fine tunning

