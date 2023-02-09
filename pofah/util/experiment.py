import pathlib
import os
import pofah.util.config as co
import pofah.path_constants.experiment_dict as exdi
import pofah.util.utility_fun as utfu

class Experiment():

    def __init__(self, run_n=0, param_dict={}, path_dict=exdi.path_dict):
        self.run_n = run_n
        self.run_dir = 'run_' + str(self.run_n)
        self.param_dict = {**param_dict, '$run$': self.run_dir}
        self.path_dict = path_dict
        self.fig_dir = os.path.join(co.config['fig_dir'], self.run_dir)
        self.fig_dir_event = os.path.join(self.fig_dir,'analysis_event')
        self.model_analysis_dir = os.path.join(co.config['model_analysis_base_dir'], self.run_dir)


    def setup(self, fig_dir=False, result_dir=False, tensorboard_dir=False, model_dir=False, model_dir_qr=False, analysis_dir=False, analysis_dir_qr=False, model_analysis_dir=True, model_comparison_dir=False):

        if fig_dir:
            pathlib.Path(self.fig_dir).mkdir(parents=True, exist_ok=True)
            self.fig_dir_img = os.path.join(self.fig_dir,'analysis_image')
            pathlib.Path(self.fig_dir_img).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.fig_dir_event).mkdir(parents=True, exist_ok=True)

        if tensorboard_dir:
            self.tensorboard_dir = os.path.join(co.config['tensorboard_dir'], self.run_dir)
            pathlib.Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)

        # model paths VAE
        if model_dir:
            self.model_dir = utfu.multi_replace(text=self.path_dict['model_dir'], repl_dict=self.param_dict)
            pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # model paths QR
        if model_dir_qr:
            self.model_dir_qr = utfu.multi_replace(text=self.path_dict['model_dir_qr'], repl_dict={**self.param_dict, '$run$': str(self.run_n)}) # qr paths supporting run_x format where x is value instead of entire 'run_x' string
            pathlib.Path(self.model_dir_qr).mkdir(parents=True, exist_ok=True)            

        # analysis paths VAE
        if analysis_dir:
            self.analysis_dir_fig = utfu.multi_replace(text=self.path_dict['analysis_base_dir_fig'], repl_dict=self.param_dict)
            self.analysis_dir_bin_count = utfu.multi_replace(text=self.path_dict['analysis_base_dir_bin_count'], repl_dict=self.param_dict)
            pathlib.Path(self.analysis_dir_fig).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.analysis_dir_bin_count).mkdir(parents=True, exist_ok=True)

        if analysis_dir_qr:
            analysis_base_dir_qr = utfu.multi_replace(text=self.path_dict['analysis_base_dir_qr'], repl_dict={**self.param_dict, '$run$': str(self.run_n)})
            self.analysis_dir_qr_mjj = os.path.join(analysis_base_dir_qr, 'mjj_spectra')
            self.analysis_dir_qr_cuts = os.path.join(analysis_base_dir_qr, 'qr_cuts')
            pathlib.Path(self.analysis_dir_qr_mjj).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.analysis_dir_qr_cuts).mkdir(parents=True, exist_ok=True)


        if model_analysis_dir:
            self.model_analysis_dir_roc = os.path.join(self.model_analysis_dir, 'roc')
            self.model_analysis_dir_loss = os.path.join(self.model_analysis_dir, 'loss')
            
            pathlib.Path(self.model_analysis_dir_roc).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.model_analysis_dir_loss).mkdir(parents=True, exist_ok=True)
            
        if model_comparison_dir:
            self.model_comparison_dir = utfu.multi_replace(text=self.path_dict['model_comparison_dir'], repl_dict=self.param_dict)
            self.model_comparison_dir_roc = os.path.join(self.model_comparison_dir, 'roc')
            self.model_comparison_dir_loss = os.path.join(self.model_comparison_dir, 'loss')
            pathlib.Path(self.model_comparison_dir_roc).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.model_comparison_dir_loss).mkdir(parents=True, exist_ok=True)
        
        return self

