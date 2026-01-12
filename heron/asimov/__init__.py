import os
import configparser
import glob
import shutil
import sys

import asimov.pipeline
from asimov import config
import htcondor
from asimov.utils import set_directory
from ..utils import make_metafile

# Handle importlib.resources compatibility across Python versions
if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files

class MetaPipeline(asimov.pipeline.Pipeline):

    def build_dag(self, dryrun=False, psds=None, user=None, clobber_psd=False):
        """
        Create a condor submission description.
        """
        name = self.production.name  # meta['name']
        ini = self.production.event.repository.find_prods(name, self.category)[0]

        # Construct the command arguments properly
        arguments = " ".join(self._pipeline_arguments) + f" {ini}"

        description = {
            "executable": f"{os.path.join(config.get('pipelines', 'environment'), 'bin', self._pipeline_command)}",
            "arguments": arguments,
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

        if not dryrun:
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
            self.production.status = "running"
        else:
            self.logger.info(f"Dry run: would submit job with description: {description}")
            self.clusterid = None

    def submit_dag(self, dryrun=False):
        """Submit the job to the cluster. For MetaPipeline, submission happens in build_dag."""
        if hasattr(self, 'clusterid') and self.clusterid is not None:
            return self.clusterid
        else:
            return None

class InjectionPipeline(MetaPipeline):
    name = "heron injection"
    config_template = str(files("heron.asimov") / "heron_template.yml")
    _pipeline_command = "heron"
    _pipeline_arguments = ["injection", "--settings"]

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
    config_template = str(files("heron.asimov") / "heron_template.yml")

    _pipeline_command = "heron"
    _pipeline_arguments = ["inference", "--settings"]

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
