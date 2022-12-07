# Generate OptimizedConfiguration for GAMESS using ANI

 Optimizes the geometry of a molecule using the ANI neural potential (`ani2x`, trained upon `wB97x/6-31G*` for C, H, N, O, S, F, and Cl)) and adds the optimized geometry to an input file for GAMESS (e.g. `nvcr.io/hpc/gamess:17.09-r2-libcchem`).

## Quick links

- [Getting started](#getting-started)
- [Development](#development)
- [Help and Support](#help-and-support)
- [Contributing](#contributing)
- [License](#license)

## Getting started

1. Get access to an environment hosting a deployment of the Simulation Toolkit for Scientific Discovery ([ST4SD](https://st4sd.github.io/overview)). 
2. [Push](https://st4sd.github.io/overview/creating-a-parameterised-package#adding-a-parameterised-package-to-a-registry) the experiment to your ST4SD deployment.
   1. You can find the JSON representation of the parameterised package in ['parameterised-packages/configuration-generator-ani.json'](parameterised-packages/configuration-generator-ani.json).
3. [Start](https://st4sd.github.io/overview/running-workflows-on-openshift#running-a-virtual-experiment) your experiment
4. Download the [outputs](https://st4sd.github.io/overview/running-workflows-on-openshift#retrieving-the-outputs-of-a-virtual-experiment-instance) of your experiment run after your run terminates


### Adding the experiment to ST4SD

This will add a basic parameterised package for this experiment called `configuration-generator-ani`

```python
api.api_experiment_push({"base": {"packages": [{"config": {"manifestPath": "manifest.yaml", "path": "ani-surrogate.yaml"}, "dependencies": {"imageRegistries": []}, "name": "main", "source": {"git": {"location": {"tag": "1.0.0", "url": "https://github.com/st4sd/configuration-generator-ani.git"}}}}]}, "metadata": {"package": {"description": "Surrogate that optimizes the geometry of a molecule using the ANI neural potential (ani2x, functional: vWB97x) and adds it to a GAMESS molecule.inp file", "keywords": ["smiles", "computational chemistry", "geometry-optimization", "gamess", "surrogate"], "maintainer": "https://github.com/Jammyzx1", "name": "configuration-generator-ani", "tags": ["1.0.0"]}}, "parameterisation": {"executionOptions": {"data": [], "platform": [], "runtime": {"args": [], "resources": {}}, "variables": [{"name": "molecule_index"}, {"name": "n_conformers"}, {"name": "max_iterations"}]}, "presets": {"data": [], "environmentVariables": [], "platform": "openshift", "runtime": {"args": ["--registerWorkflow=yes"], "resources": {}}, "variables": [{"name": "ani_model", "value": "ani2x"}, {"name": "optimizer", "value": "bfgs"}, {"name": "functional", "value": "wB97X"}]}}})
```

You can create your own parameterised packages for these experiments. See [here](https://st4sd.github.io/overview/creating-a-parameterised-package) for more information about how to do this. This repo contains files with the above json in easy to edit form you can use as a basis. 

## Experiment inputs

The experiment requires an input CSV file, called `input_smiles.csv`, with columns `label` and `smiles`. The label column should contain unique integers (e.g. the row number). The `smiles` column should contain the [SMILE](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) representation of the input molecules. The file can contain other columns - these will be ignored. 

Example:

```
label,smiles
0,O=S(=O)([O-])c1c(C(F)(F)F)cc(C(F)(F)F)cc1C(F)(F)F.Cc1cc(OC(C)(C)C)cc(C)c1[S+](c1ccccc1)c1ccccc1
1,O=S(=O)([O-])c1ccc(C(F)(F)F)cc1C(F)(F)F.Cc1ccc([S+](c2ccccc2)c2ccccc2)c(C)c1
```

An example payload for the experiment can be created as follows:

```python
smiles_data = pd.read_csv('smiles_data.csv')[['label', 'SMILES']]
payload = {
    "inputs": [{
       "content": smiles_data.to_csv(index=False),
       "filename": "input_smiles.csv"
   }]
}
```


## Development

1. Fork this repository. You will find a virtual experiments in this repository. 
2. Modify the code
3. Push your code to your forked github repository. Then follow the getting started instructions above.

**Note**: Remember to update your `parameterised package` payload so that it points to your forked GitHub repository.

## Help and Support

Please feel free to reach out to one of the maintainers listed in the [MAINTAINERS.md](MAINTAINERS.md) page.

## Contributing

We always welcome external contributions. Please see our [guidance](CONTRIBUTING.md) for details on how to do so.

## License

This project is licensed under the Apache 2.0 license. Please [see details here](LICENSE.md).
