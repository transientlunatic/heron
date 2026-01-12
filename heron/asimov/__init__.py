import importlib
import pkg_resources
import os
import configparser
import glob
import shutil

import asimov.pipeline
from asimov import config
import htcondor
from asimov.utils import set_directory
from ..utils import make_metafile

class MetaPipeline(asimov.pipeline.Pipeline):

    def build_dag(self, dryrun=False):
        """
        Create a condor submission description.
        """
        name = self.production.name  # meta['name']
        ini = self.production.event.repository.find_prods(name, self.category)[0]
        description = {
            "executable": f"{os.path.join(config.get('pipelines', 'environment'), 'bin', self._pipeline_command)}",
            "arguments": f"{self._pipeline_arguments[0]} --settings {ini}",
            "output": f"{name.replace(' ', '_')}.out",
            "error": f"{name.replace(' ', '_')}.err",
            "log": f"{name.replace(' ', '_')}.log",
            "request_gpus": 1,
            "batch_name": f"{self.name}/{self.production.event.name}/{name}",
        }

        job = htcondor.Submit(description)
        os.makedirs(self.production.rundir, exist_ok=True)
        with set_directory(self.production.rundir):
            with open(f"{name}.sub", "w") as subfile:
                subfile.write(job.__str__())

        with set_directory(self.production.rundir):
            try:
                schedulers = htcondor.Collector().locate(
                    htcondor.DaemonTypes.Schedd, config.get("condor", "scheduler")
                )
            except configparser.NoOptionError:
                schedulers = htcondor.Collector().locate(htcondor.DaemonTypes.Schedd)
            schedd = htcondor.Schedd(schedulers)
            with schedd.transaction() as txn:
                cluster_id = job.queue(txn)

        self.production.job_id = int(cluster_id)
        self.clusterid = cluster_id

    def submit_dag(self, dryrun=False):
        return self.clusterid

class InjectionPipeline(MetaPipeline):
    name = "heron injection"
    config_template = importlib.resources.files("heron.asimov") / "heron_template.yml"
    _pipeline_command = "heron"
    _pipeline_arguments = ["injection"]

    def detect_completion(self):

        self.logger.info("Checking for completion.")
        assets = self.collect_assets()
        if "frames" in assets:
            posterior = assets["frames"]
            self.logger.info("Frames detected, job complete.")
            return True
        else:
            self.logger.info("Frame generation job completion was not detected.")
            return False

    def after_completion(self):
        self.production.status = "uploaded"
        self.production.event.update_data()
        
    def collect_assets(self):
        """
        Collect the assets for this job.
        """

        outputs = {}
        if os.path.exists(os.path.join(self.production.rundir, "frames")):
            results_dir = glob.glob(os.path.join(self.production.rundir, "frames", "*"))
            frames = {}

            for frame in results_dir:
                ifo = frame.split("/")[-1].split("-")[0]
                frames[ifo] = frame

            outputs["frames"] = frames

            self.production.event.meta['data']['data files'] = frames

        if os.path.exists(os.path.join(self.production.rundir, "psds")):
            results_dir = glob.glob(os.path.join(self.production.rundir, "psds", "*"))
            frames = {}

            for frame in results_dir:
                ifo = frame.split("/")[-1].split(".")[0]
                frames[ifo] = frame

            outputs["psds"] = frames
            
        self.production.event.update_data()
        return outputs


class Pipeline(MetaPipeline):
    """
    An asimov pipeline for heron.
    """

    name = "heron"
    config_template = importlib.resources.files("heron.asimov") / "heron_template.yml"

    _pipeline_command = "heron"
    _pipeline_arguments = ["inference"]

    def detect_completion(self):

        self.logger.info("Checking for completion.")
        assets = self.collect_assets()
        if "posterior" in assets:
            posterior = assets["posterior"]
            self.logger.info("Posterior samples detected, job complete.")
            return True
        else:
            self.logger.info("Datafind job completion was not detected.")
            return False

    def after_completion(self):
        posterior = self.collect_assets()["posterior"]
        datfile = os.path.join(self.production.rundir, self.production.name, "result.dat")
        if not os.path.exists(datfile):
            make_metafile(
                posterior,
                datfile,
            )
        post_pipeline = asimov.pipeline.PESummaryPipeline(production=self.production)
        self.logger.info("Job has completed. Running PE Summary.")
        cluster = post_pipeline.submit_dag()
        self.production.meta["job id"] = int(cluster)
        self.production.status = "processing"
        self.production.event.update_data()

    def collect_assets(self):
        """
        Collect the assets for this job.
        """

        outputs = {}
        files = {"posterior": os.path.join(self.production.rundir, self.production.name, self.production.name, "result.hdf5")}
        for name, data_file in files.items():
            if os.path.exists(data_file):
                outputs[name] = data_file

        self.production.event.update_data()
        return outputs

    def samples(self, absolute=False):
        """
        Return the PESummary ready samples for this job
        """
        if absolute:
            rundir = os.path.abspath(self.production.rundir)
        else:
            rundir = self.production.rundir

        return [os.path.join(rundir, self.production.name, "result.dat")]

    def html(self):
        """Return the HTML representation of this pipeline."""
        pages_dir = os.path.join(self.production.event.name, self.production.name)
        pages_dir_full = os.path.join(config.get("general", "webroot"), pages_dir)
        plots_dir = os.path.join(pages_dir_full, "plots")
        os.makedirs(pages_dir_full, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        out = ""

        image_card = """<div class="col"><div class="card" style="width: 18rem;">
<img class="card-img-top" src="{0}" alt="Card image cap">
  <div class="card-body">
    <p class="card-text">{1}</p>
  </div>
</div></div>
        """
        os.makedirs(pages_dir, exist_ok=True)

        for png_file in glob.glob(
            f"{self.production.rundir}/{self.production.name}/*.png"
        ):
            name = png_file.split("/")[-1]
            shutil.copy(png_file, os.path.join(plots_dir, name))
        out += """<div class="asimov-pipeline heron row">"""
        if self.production.status in {"running", "stuck"}:
            out += image_card.format(
                f"{pages_dir}/plots/trace.png",
                "Trace plot",
            )
            out += image_card.format(
                f"{pages_dir}/plots/state.png",
                "State plot",
            )
        if self.production.status in {"finished", "uploaded"}:
            # out += (
            #     f"""<p><a href="{pages_dir}/index.html">Full Megaplot output</a></p>"""
            # )
            out += image_card.format(
                f"{pages_dir}/plots/posterior.png",
                "Posterior plot",
            )

        out += """</div>"""

        return out


class TrainingDataPipeline(MetaPipeline):
    """
    Pipeline for generating training data for GPR models.
    """

    name = "heron training data"
    config_template = importlib.resources.files("heron.asimov") / "heron_training_data_template.yml"
    _pipeline_command = "heron"
    _pipeline_arguments = ["training-data"]

    def detect_completion(self):
        """
        Check for completion by looking for the HDF5 training data file.
        """
        self.logger.info("Checking for training data generation completion.")
        assets = self.collect_assets()
        if "training_data" in assets:
            self.logger.info("Training data file detected, job complete.")
            return True
        else:
            self.logger.info("Training data generation not yet complete.")
            return False

    def after_completion(self):
        """
        Set status to uploaded after completion.
        """
        self.production.status = "uploaded"
        self.production.event.update_data()

    def collect_assets(self):
        """
        Collect training data assets.

        Returns
        -------
        dict
            Dictionary with 'training_data' and 'training_plots' keys
        """
        outputs = {}

        # Look for HDF5 training data file
        training_file = os.path.join(
            self.production.rundir,
            f"{self.production.name}_training.h5"
        )
        if os.path.exists(training_file):
            outputs["training_data"] = training_file

            # Store in event metadata for downstream access
            if 'training' not in self.production.event.meta:
                self.production.event.meta['training'] = {}
            self.production.event.meta['training']['data_file'] = training_file

        # Look for plots directory
        plots_dir = os.path.join(self.production.rundir, "plots")
        if os.path.exists(plots_dir):
            outputs["training_plots"] = plots_dir

        self.production.event.update_data()
        return outputs

    def html(self):
        """Generate HTML report for training data generation."""
        pages_dir = os.path.join(self.production.event.name, self.production.name)
        pages_dir_full = os.path.join(config.get("general", "webroot"), pages_dir)
        plots_dir = os.path.join(pages_dir_full, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        out = '<div class="asimov-pipeline heron-training-data">'

        # Training data summary
        out += '<h4>Training Data Generation</h4>'

        # Show configuration
        if 'waveform_source' in self.production.meta:
            source = self.production.meta['waveform_source']
            approx_name = source.get("approximant", source.get("catalog_type", "Unknown"))
            out += f'<p><strong>Waveform Source:</strong> {approx_name}</p>'

        # Show parameter space
        if 'parameter_space' in self.production.meta:
            params = self.production.meta['parameter_space']
            out += '<details><summary>Parameter Space</summary>'
            out += '<ul>'
            if 'fixed' in params:
                for key, val in params['fixed'].items():
                    out += f'<li><strong>{key}:</strong> {val} (fixed)</li>'
            if 'varied' in params:
                for key, val in params['varied'].items():
                    lower = val.get('lower', '?')
                    upper = val.get('upper', '?')
                    out += f'<li><strong>{key}:</strong> {lower} - {upper}</li>'
            out += '</ul></details>'

        # Copy diagnostic plots to web directory
        source_plots = os.path.join(self.production.rundir, "plots")
        if os.path.exists(source_plots):
            for png_file in glob.glob(os.path.join(source_plots, "*.png")):
                name = png_file.split("/")[-1]
                shutil.copy(png_file, os.path.join(plots_dir, name))

        # Display plots based on status
        image_card = """<div class="col"><div class="card" style="width: 18rem;">
            <img class="card-img-top" src="{0}" alt="{1}">
            <div class="card-body"><p class="card-text">{1}</p></div>
        </div></div>"""

        if self.production.status in {"finished", "uploaded"}:
            out += '<div class="row">'

            # Parameter space coverage plot
            if os.path.exists(os.path.join(plots_dir, "parameter_space.png")):
                out += image_card.format(
                    f"{pages_dir}/plots/parameter_space.png",
                    "Parameter space coverage"
                )

            # Example waveforms
            if os.path.exists(os.path.join(plots_dir, "waveform_samples.png")):
                out += image_card.format(
                    f"{pages_dir}/plots/waveform_samples.png",
                    "Sample waveforms from manifold"
                )

            # Waveform manifold heatmap
            if os.path.exists(os.path.join(plots_dir, "manifold_heatmap.png")):
                out += image_card.format(
                    f"{pages_dir}/plots/manifold_heatmap.png",
                    "Waveform manifold (time vs parameters)"
                )

            out += '</div>'

        # Dataset statistics
        training_file = os.path.join(self.production.rundir, f"{self.production.name}_training.h5")
        if os.path.exists(training_file):
            try:
                import h5py
                with h5py.File(training_file, 'r') as f:
                    if 'training data' in f:
                        groups = list(f['training data'].keys())
                        n_waveforms = 0
                        for g in groups:
                            if 'data' in f[f'training data/{g}']:
                                n_waveforms += len(f[f'training data/{g}/data'])

                        out += f'<p><strong>Dataset:</strong> {n_waveforms} waveforms across {len(groups)} groups</p>'
            except Exception as e:
                self.logger.warning(f"Could not read HDF5 stats: {e}")

            out += f'<p><strong>File:</strong> <code>{training_file}</code></p>'

        out += '</div>'
        return out


class GPRTrainingPipeline(MetaPipeline):
    """
    Pipeline for training GPR models from training data.
    """

    name = "heron gpr training"
    config_template = importlib.resources.files("heron.asimov") / "heron_gpr_training_template.yml"
    _pipeline_command = "heron"
    _pipeline_arguments = ["train-gpr"]

    def get_training_data_file(self):
        """
        Get training data file from dependency using bilby-style pattern.

        Returns
        -------
        str or None
            Path to training data HDF5 file if available
        """
        if self.production.dependencies:
            productions = {p.name: p for p in self.production.event.productions}
            for previous_job in self.production.dependencies:
                if previous_job in productions:
                    assets = productions[previous_job].pipeline.collect_assets()
                    if "training_data" in assets:
                        return assets['training_data']
        return None

    def detect_completion(self):
        """
        Check for completion by verifying model states exist in training data file.
        """
        self.logger.info("Checking for GPR training completion.")

        # Get training data file
        training_data_file = self.get_training_data_file()
        if not training_data_file:
            self.logger.warning("No training data file found from dependencies")
            return False

        if not os.path.exists(training_data_file):
            self.logger.warning(f"Training data file not found: {training_data_file}")
            return False

        # Check if model states exist
        try:
            import h5py
            with h5py.File(training_data_file, 'r') as f:
                if 'model states' in f:
                    model_name = self.production.meta.get('model_name', 'gpr_model')
                    expected_states = [f"{model_name}_plus", f"{model_name}_cross"]

                    has_states = all(
                        state_name in f['model states']
                        for state_name in expected_states
                    )

                    if has_states:
                        self.logger.info("Model states detected, training complete.")
                        return True
                    else:
                        self.logger.info(f"Missing model states: {expected_states}")
                        return False
        except Exception as e:
            self.logger.warning(f"Error checking model states: {e}")
            return False

        return False

    def after_completion(self):
        """
        Set status to uploaded after completion.
        """
        self.production.status = "uploaded"
        self.production.event.update_data()

    def collect_assets(self):
        """
        Collect GPR training assets.

        Returns
        -------
        dict
            Dictionary with 'model_states' and 'training_plots' keys
        """
        outputs = {}

        # Get training data file (which now contains model states)
        training_data_file = self.get_training_data_file()
        if training_data_file and os.path.exists(training_data_file):
            outputs["model_states"] = training_data_file

            # Store in event metadata for downstream access
            if 'gpr_models' not in self.production.event.meta:
                self.production.event.meta['gpr_models'] = {}
            self.production.event.meta['gpr_models']['trained_model'] = training_data_file

        # Look for plots directory
        plots_dir = os.path.join(self.production.rundir, "plots")
        if os.path.exists(plots_dir):
            outputs["training_plots"] = plots_dir

        self.production.event.update_data()
        return outputs

    def html(self):
        """Generate HTML report for GPR training."""
        pages_dir = os.path.join(self.production.event.name, self.production.name)
        pages_dir_full = os.path.join(config.get("general", "webroot"), pages_dir)
        plots_dir = os.path.join(pages_dir_full, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        out = '<div class="asimov-pipeline heron-gpr-training">'

        # Training summary
        out += '<h4>GPR Model Training</h4>'

        # Show configuration
        if 'model_name' in self.production.meta:
            model_name = self.production.meta['model_name']
            out += f'<p><strong>Model:</strong> {model_name}</p>'

        if 'hyperparameters' in self.production.meta:
            hyperparams = self.production.meta['hyperparameters']
            out += '<details><summary>Training Hyperparameters</summary>'
            out += '<ul>'
            for key, val in hyperparams.items():
                out += f'<li><strong>{key}:</strong> {val}</li>'
            out += '</ul></details>'

        # Copy diagnostic plots to web directory
        source_plots = os.path.join(self.production.rundir, "plots")
        if os.path.exists(source_plots):
            for png_file in glob.glob(os.path.join(source_plots, "*.png")):
                name = png_file.split("/")[-1]
                shutil.copy(png_file, os.path.join(plots_dir, name))

        # Display plots based on status
        image_card = """<div class="col"><div class="card" style="width: 18rem;">
            <img class="card-img-top" src="{0}" alt="{1}">
            <div class="card-body"><p class="card-text">{1}</p></div>
        </div></div>"""

        if self.production.status in {"running", "stuck"}:
            out += '<div class="row">'

            # Show latest training diagnostics
            if os.path.exists(os.path.join(plots_dir, "training_loss.png")):
                out += image_card.format(
                    f"{pages_dir}/plots/training_loss.png",
                    "Training loss evolution"
                )

            if os.path.exists(os.path.join(plots_dir, "hyperparameters.png")):
                out += image_card.format(
                    f"{pages_dir}/plots/hyperparameters.png",
                    "Hyperparameter evolution"
                )

            out += '</div>'

        if self.production.status in {"finished", "uploaded"}:
            out += '<div class="row">'

            # Final training curves
            if os.path.exists(os.path.join(plots_dir, "training_loss.png")):
                out += image_card.format(
                    f"{pages_dir}/plots/training_loss.png",
                    "Final training loss"
                )

            if os.path.exists(os.path.join(plots_dir, "hyperparameters.png")):
                out += image_card.format(
                    f"{pages_dir}/plots/hyperparameters.png",
                    "Final hyperparameters"
                )

            # Prediction validation plots
            if os.path.exists(os.path.join(plots_dir, "predictions_plus.png")):
                out += image_card.format(
                    f"{pages_dir}/plots/predictions_plus.png",
                    "Plus polarization predictions"
                )

            if os.path.exists(os.path.join(plots_dir, "predictions_cross.png")):
                out += image_card.format(
                    f"{pages_dir}/plots/predictions_cross.png",
                    "Cross polarization predictions"
                )

            out += '</div>'

        # Model statistics
        training_data_file = self.get_training_data_file()
        if training_data_file and os.path.exists(training_data_file):
            try:
                import h5py
                with h5py.File(training_data_file, 'r') as f:
                    if 'model states' in f:
                        model_name = self.production.meta.get('model_name', 'gpr_model')
                        states = [s for s in f['model states'].keys() if model_name in s]
                        out += f'<p><strong>Trained models:</strong> {len(states)} polarization(s)</p>'
            except Exception as e:
                self.logger.warning(f"Could not read model states: {e}")

            out += f'<p><strong>Model file:</strong> <code>{training_data_file}</code></p>'

        out += '</div>'
        return out
