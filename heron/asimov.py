import importlib
import os
import configparser

import asimov.pipeline
from asimov import config
import htcondor
import yaml
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
        name = self.production.name #meta['name']
        ini = self.production.event.repository.find_prods(name,
                                                          self.category)[0]
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
                schedulers = htcondor.Collector().locate(htcondor.DaemonTypes.Schedd, config.get("condor", "scheduler"))
            except configparser.NoOptionError:
                schedulers = htcondor.Collector().locate(htcondor.DaemonTypes.Schedd)
            schedd = htcondor.Schedd(schedulers)
            with schedd.transaction() as txn:
                cluster_id = job.queue(txn)

        self.clusterid = cluster_id

    def submit_dag(self, dryrun=False):
        return self.clusterid
    
def submit_description():
    schedd = htcondor.Schedd(schedulers)
    with schedd.transaction() as txn:   
        cluster_id = job.queue(txn)
    return cluster_id
