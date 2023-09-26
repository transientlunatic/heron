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
from .utils import make_metafile

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
    config_template = importlib.resources.files("heron") / "heron_template.yml"
    _pipeline_command = "heron"
    _pipeline_arguments = ["injection"]

class Pipeline(MetaPipeline):
    """
    An asimov pipeline for heron.
    """

    name = "heron"
    config_template = importlib.resources.files("heron") / "heron_template.yml"

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
