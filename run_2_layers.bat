REM run 1 layer over the whole data until convergence
python enwik8_run.py --name=2_layers --num_layers=2 --sweeps=0 --restore=True --units_num=1500 --time_steps=50 --drop_output_init=0.1 --drop_output_step=0.05 --drop_state_init=0.1 --drop_state_step=0.05 --drop_emb=0.2 --batch_size=100 --train_size=95000000 --valid_size=5000000

REM run the layer over a small coherent part of the data for fine tunning

