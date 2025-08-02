import subprocess

subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.05','--sigma=0.5','--dt0=0.005','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=milstein',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.05','--sigma=0.1','--dt0=0.005','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=milstein',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.05','--sigma=0.4','--dt0=0.005','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=milstein',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.05','--sigma=0.2','--dt0=0.005','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=milstein',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.05','--sigma=0.3','--dt0=0.005','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=milstein',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.025','--sigma=0.1','--dt0=0.005','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=milstein',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])
#
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.05','--sigma=0.1','--dt0=0.1','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=reversible_heun',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.05','--sigma=0.1','--dt0=0.05','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=reversible_heun',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.05','--sigma=0.1','--dt0=0.01','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=reversible_heun',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.05','--sigma=0.1','--dt0=0.001','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=reversible_heun',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])

subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.05','--sigma=0.1','--dt0=0.005','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=euler_heun',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.05','--sigma=0.1','--dt0=0.005','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=heun',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.05','--sigma=0.1','--dt0=0.005','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=midpoint',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=van_der_pol',
                '--scale=0.05','--sigma=0.1','--dt0=0.005','--hidden_size_drift=64','--depth_drift=2',
                '--hidden_size_diff=64','--depth_diff=2','--n_blocks=2','--partial=False','--solver_name=milstein',
                '--noise_type=one','--patch_size=10','--use_normalization=True','--norm_method=minmax',
                '--calculation=True','--eval_and_plot=False'])

subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=fhn',
                '--scale=0.05', '--dt0=0.005','--hidden_size_drift=64','--depth_drift=3','--hidden_size_diff=16',
                '--depth_diff=1', '--n_blocks=2','--partial=True','--solver_name=milstein','--noise_type=diag',
                '--patch_size=5','--use_normalization=True','--norm_method=minmax',
                '--batch_norm_last=True', '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=fhn',
                '--scale=0.025', '--dt0=0.005','--hidden_size_drift=64','--depth_drift=3','--hidden_size_diff=16',
                '--depth_diff=1', '--n_blocks=2','--partial=True','--solver_name=milstein','--noise_type=diag',
                '--patch_size=5','--use_normalization=True','--norm_method=minmax',
                '--batch_norm_last=True', '--calculation=True','--eval_and_plot=False'])
#
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=lotka_volterra',
                '--scale=0.05','--dt0=0.005','--hidden_size_drift=64','--depth_drift=3','--hidden_size_diff=64',
                '--depth_diff=2', '--n_blocks=2', '--batch_norm=True', '--batch_norm_last=False', '--partial=False',
                '--solver_name=milstein', '--noise_type=diag', '--patch_size=5','--use_normalization=True',
                '--norm_method=minmax', '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=synthetic','--data_set=lotka_volterra',
                '--scale=0.025','--dt0=0.005','--hidden_size_drift=64','--depth_drift=3','--hidden_size_diff=64',
                '--depth_diff=2', '--n_blocks=2', '--batch_norm=True', '--batch_norm_last=False', '--partial=False',
                '--solver_name=milstein', '--noise_type=diag', '--patch_size=5','--use_normalization=True',
                '--norm_method=minmax', '--calculation=True','--eval_and_plot=False'])
#
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=real_world','--data_set=acrobot_noisy',
                '--x_size=4', '--u_size=1','--dt0=0.005','--hidden_size_drift=128','--depth_drift=3','--hidden_size_diff=16',
                '--depth_diff=1','--n_blocks=2','--partial=False','--solver_name=milstein','--noise_type=one',
                '--patch_size=5','--use_normalization=True','--norm_method=minmax', '--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','NF_Stratonovich_NeuralSDE_models_test.py','--data_type=real_world','--data_set=cartpole_noisy',
                '--x_size=4', '--u_size=1','--dt0=0.005','--hidden_size_drift=128','--depth_drift=3','--hidden_size_diff=16',
                '--depth_diff=1','--n_blocks=2','--partial=False','--solver_name=milstein','--noise_type=one',
                '--patch_size=10','--use_normalization=True','--norm_method=minmax', '--calculation=True','--eval_and_plot=False'])

subprocess.run(['python3','latent_SDE_test.py','--data_type=synthetic','--data_set=van_der_pol','--scale=0.05','--sigma=0.1',
                '--hidden_size_drift=64','--depth_drift=2','--hidden_size_diff=64','--depth_diff=2','--diff_model=one',
                '--context_dim=0','--dt0=0.005','--adaptive=False','--partial=False','--patch_size=10',
                '--use_normalization=True','--norm_method=minmax','--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','latent_SDE_test.py','--data_type=synthetic','--data_set=van_der_pol','--scale=0.025','--sigma=0.1',
                '--hidden_size_drift=64','--depth_drift=2','--hidden_size_diff=64','--depth_diff=2','--diff_model=one',
                '--context_dim=0','--dt0=0.005','--adaptive=False','--partial=False','--patch_size=10',
                '--use_normalization=True','--norm_method=minmax','--calculation=True','--eval_and_plot=False'])

subprocess.run(['python3','latent_SDE_test.py','--data_type=synthetic','--data_set=fhn','--scale=0.05',
                '--hidden_size_drift=64','--depth_drift=3','--hidden_size_diff=64','--depth_diff=2','--diff_model=diag',
                '--context_dim=0','--dt0=0.005','--adaptive=False','--partial=True','--scaled=False', '--patch_size=10',
                '--use_normalization=True','--norm_method=minmax','--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','latent_SDE_test.py','--data_type=synthetic','--data_set=fhn','--scale=0.025',
                '--hidden_size_drift=64','--depth_drift=3','--hidden_size_diff=64','--depth_diff=2','--diff_model=diag',
                '--context_dim=0','--dt0=0.005','--adaptive=False','--partial=True','--scaled=False', '--patch_size=10',
                '--use_normalization=True','--norm_method=minmax','--calculation=True','--eval_and_plot=False'])

subprocess.run(['python3','latent_SDE_test.py','--data_type=synthetic','--data_set=lotka_volterra','--scale=0.05',
                '--hidden_size_drift=64','--depth_drift=3','--hidden_size_diff=64','--depth_diff=2','--diff_model=diag',
                '--context_dim=0','--dt0=0.005','--adaptive=False','--partial=True','--patch_size=10',
                '--use_normalization=True','--norm_method=minmax','--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','latent_SDE_test.py','--data_type=synthetic','--data_set=lotka_volterra','--scale=0.025',
                '--hidden_size_drift=64','--depth_drift=3','--hidden_size_diff=64','--depth_diff=2','--diff_model=diag',
                '--context_dim=0','--dt0=0.005','--adaptive=False','--partial=True','--patch_size=10',
                '--use_normalization=True','--norm_method=minmax','--calculation=True','--eval_and_plot=False'])

subprocess.run(['python3','latent_SDE_test.py','--data_type=real_world','--data_set=acrobot_noisy','--u_size=1',
                '--hidden_size_drift=128','--depth_drift=3','--hidden_size_diff=64','--depth_diff=2','--context_dim=0',
                '--dt0=0.005','--adaptive=False','--partial=True','--patch_size=10',
                '--use_normalization=True','--norm_method=minmax','--calculation=True','--eval_and_plot=False'])
subprocess.run(['python3','latent_SDE_test.py','--data_typ=real_world','--data_set=cartpole_noisy','--u_size=1',
                '--hidden_size_drift=128','--depth_drift=3','--hidden_size_diff=64','--depth_diff=2','--context_dim=0',
                '--dt0=0.005','--adaptive=False','--partial=True','--patch_size=10',
                '--use_normalization=True','--norm_method=minmax','--calculation=True','--eval_and_plot=False'])