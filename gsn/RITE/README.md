# README

## This is a brief tutorial for a prototype Windows release of the [RITE tool](https://github.com/ge-high-assurance/RITE) with a functionality to generate Overarching Properties-based assurance cases in the Goal Structuring Notation (GSN) from the [RACK curated database](https://github.com/ge-high-assurance/RACK)


### Prerequsites:
1. The prototype RITE tool release with the OP-based GSN functionality available at https://github.com/ge-high-assurance/OPAI/releases/tag/gsn_v1
2. RACK docker (find tutorial [here](https://github.com/ge-high-assurance/RACK/wiki/02-Run-a-RACK-Box-container))
3. The ingestion package provided at https://github.com/ge-high-assurance/OPAI/tree/main/gsn/ingestion_packages/assurance_v1
4. Java 17
**Note**: Kindly look at the [RITE Wiki](https://github.com/ge-high-assurance/RITE/wiki) for general details about RITE


### Steps for using the tool:
1. Start a RACK docker container
2. Unzip the release available at https://github.com/ge-high-assurance/OPAI/releases/tag/gsn_v1
3. Click on RITE

 ![1](https://github.com/ge-high-assurance/OPAI/assets/66636651/9baf2fe5-e932-4697-9e80-969b4f82bdd6)

4. Select a workspace 

![2](https://github.com/ge-high-assurance/OPAI/assets/66636651/6f2ead65-a9df-4a09-b36c-698db4170f5f)

5. Clear RACK 

![3](https://github.com/ge-high-assurance/OPAI/assets/66636651/7786bfd5-7beb-4927-9af2-521ab8964a8f) 

![4](https://github.com/ge-high-assurance/OPAI/assets/66636651/6fabb92e-5515-479e-bbe9-ec8a6a01edd7)

6. Import the ingestion package 

![5](https://github.com/ge-high-assurance/OPAI/assets/66636651/a914661d-174d-40ae-a01c-80d38ef40c5c) ![6](https://github.com/ge-high-assurance/OPAI/assets/66636651/37edb481-4caa-4351-a243-e8d514950014)

7. Ingest the package 

![7](https://github.com/ge-high-assurance/OPAI/assets/66636651/b45c487e-11fe-43db-be17-9f5180e77d12) ![8](https://github.com/ge-high-assurance/OPAI/assets/66636651/006b66ad-743a-4d4b-84fd-cca7418b11df)

8. Launch the Op GSN tool 

![9](https://github.com/ge-high-assurance/OPAI/assets/66636651/954ee4b3-8770-4168-90ab-6828fbbc61a0)

9. Specify a directory to store the outputs 

![10](https://github.com/ge-high-assurance/OPAI/assets/66636651/04e43aa9-92dd-458e-b983-5c9c868be592) 
and click `Generate`.

10. The interactive tool will now populate with the GSN details. Click on different elements to drill down deepr into the GSN tree. The SVG GSN artifacts can befound in the directory that was specified. 

![11](https://github.com/ge-high-assurance/OPAI/assets/66636651/2df6b349-5a14-47bb-874f-ff3a477f8df8)











