import importlib
import os
import configparser
import glob
import shutil

import asimov.pipeline
from asimov import config
import htcondor
from asimov.utils import set_directory


class Pipeline(asimov.pipeline.Pipeline):
    """
    An asimov pipeline for heron.
    """

    name = "heron"
    config_template = importlib.resources.path("heron", "heron_template.yml")
    _pipeline_command = "heron"

    def build_dag(self, dryrun=False):
        """
        Create a condor submission description.
        """
        name = self.production.name  # meta['name']
        ini = self.production.event.repository.find_prods(name, self.category)[0]
        description = {
            "executable": f"{os.path.join(config.get('pipelines', 'environment'), 'bin', self._pipeline_command)}",
            "arguments": f"inference --settings {ini}",
            "output": f"{name}.out",
            "error": f"{name}.err",
            "log": f"{name}.log",
            "request_gpus": 1,
            "batch_name": f"heron/{name}",
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

    def detect_completion(self):

        self.logger.info("Checking for completion.")
        assets = self.collect_assets()
        if "posterior" in assets:
            posterior = assets["posterior"]
            if len(list(frames.values())) > 0:
                self.logger.info("Posterior samples detected, job complete.")
                return True
            else:
                self.logger.info("Datafind job completion was not detected.")
                return False
        else:
            return False

    def after_completion(self):
        self.production.status = "uploaded"

    def collect_assets(self):
        """
        Collect the assets for this job.
        """

        outputs = {}
        files = {"posterior": "result.json"}
        for name, data_file in files.items():
            if os.path.exists(data_file):
                outputs[name] = data_file

        self.production.event.update_data()
        return outputs

    def html(self):
        """Return the HTML representation of this pipeline."""
        pages_dir = os.path.join(self.production.event.name, self.production.name)
        pages_dir_full = os.path.join(config.get("general", "webroot"), pages_dir)

        out = ""

        image_card = """<div class="card" style="width: 18rem;">
<img class="card-img-top" src="{0}" alt="Card image cap">
  <div class="card-body">
    <p class="card-text">{1}</p>
  </div>
</div>
        """
        os.makedirs(pages_dir, exist_ok=True)

        for png_file in glob.glob(f"{self.production.rundir}/*.png"):
            name = png_file.split("/")[-1]
            shutil.copy(name, os.path.join(pages_dir_full, name))
        out += """<div class="asimov-pipeline heron">"""
        if self.production.status in {"running", "stuck"}:
            out += image_card.format(
                f"{pages_dir_full}/plots/trace.png",
                "Trace plot",
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
