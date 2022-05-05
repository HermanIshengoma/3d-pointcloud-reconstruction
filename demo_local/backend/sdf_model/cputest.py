#!/usr/bin/env python3

import torch
import torch.utils.data 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


import os
import json
import time
from tqdm import tqdm

# remember to add paths in model/__init__.py for new models
from model import *
# dataloader/__init__.py
#from dataloader import *


def main():
    
    test_dataset = init_dataset("acronym", specs)
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False,
    )


    model = init_model(specs["Model"], specs, len(test_dataset))
    
    if args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
    else:
        resume = None  

    checkpoint = torch.load(resume, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])



    with tqdm(test_dataloader, unit="batch") as pbar:
        for idx, data in enumerate(pbar):
            pbar.set_description(f"Testing file {idx}")
            #_, model, _, _= model.load_model(model, None, os.path.join(args.experiment_directory, "{}.pt".format(args.continue_from)))

            if args.outdir is None:
                eval_dir = os.path.join(args.exp_dir, args.resume, data['mesh_name'][0])
            else:
                eval_dir = os.path.join(args.outdir, args.resume, data['mesh_name'][0])
            model.reconstruct(model, data, eval_dir)




    # trainer = pl.Trainer(accelerator='gpu', devices=1, precision=16, resume_from_checkpoint=resume)
    # trainer.test(model=model, test_dataloaders=test_dataloader)

   
                


def init_model(model, specs, num_objects):
    if model == "MetaSdf":
        return MetaSdf(specs)
    elif model == "DeepSDF":
        return DeepSDF(specs, num_objects)
    elif model == "Siren":
        return Siren(specs)
    elif model == "NeuralPull":
        return NeuralPull(specs, num_objects)
    elif model == "SAL":
        return SAL(specs)
    elif model == "ConvOccNet":
        return ConvOccNet(specs)
    elif model == "Baseline":
        return Baseline(specs)
    elif model == "Overfit":
        return Overfit(specs)
    else:
        print("model not loaded...")

def init_dataset(dataset, specs):

    test_split_file = specs["TestSplit"]
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    if dataset == "acronym":
        from dataloader.test_acronym import TestAcronymDataset
        return TestAcronymDataset(specs["DataSource"], test_split, 16384, pc_size=specs.get("ReconPCsize",30000))

    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train DeepSDF and shapecode")
    arg_parser.add_argument(
        "--exp_dir", "-e",
        required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r",
        default=None,
        help="continue from previous saved logs, integer value or 'last'",
    )

    arg_parser.add_argument(
        "--outdir", "-o",
        default=None
    )


    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"][0])

    main()
