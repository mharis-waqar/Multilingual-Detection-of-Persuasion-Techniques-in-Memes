# Multilingual-Detection-of-Persuasion-Techniques-in-Memes
## Sem Eval 2024-Task 4
Memes are one of the most popular type of content used in an online disinformation campaign. They are mostly effective on social media platforms, since there they can easily reach a large number of users. Memes in a disinformation campaign achieve their goal of influencing the users through a number of rhetorical and psychological techniques, such as causal oversimplification, name calling, smear.
The goal of the shared task is to build models for identifying such techniques in the textual content of a meme only (one subtask) and in a multimodal setting in which both the textual and the visual content are to be analysed together (two subtasks).

## Technical Description

Following subtasks were defined in Sem Eval 2024 Task 4:

### Subtask 1
Given only the “textual content” of a meme, identify which of the 20 persuasion techniques (https://propaganda.math.unipd.it/semeval2024task4/definitions22.html), organized in a hierarchy, it uses. If the ancestor node of a technique is selected, only a partial reward is given. This is a hierarchical multilabel classification problem. You can find a view of the hierarchy in the figure below (note that there are 22 techniques in the image, but in subtask 1 "Transfer" and ""Appeal to Strong emotion" are not present, so just picture the hierarchy without them). Full details on it are available here(https://propaganda.math.unipd.it/semeval2024task4/data/hierarchy_evaluation.html). If you need additional annotated data to solve this task, you can check the PTC corpus" (https://propaganda.math.unipd.it/ptc/) as well as the SemEval 2023 task 3 data (https://propaganda.math.unipd.it/semeval2023task3/).
### Subtask 2a
Given a meme, identify which of the 22 persuasion techniques(https://propaganda.math.unipd.it/semeval2024task4/definitions22.html), organized in a hierarchy, are used both in the textual and in the visual content of the meme (multimodal task). If the ancestor node of a technique is selected, only partial reward will be given. This is a hierarchical multilabel classification problem. You can find info on the hierarchy here.
### Subtask 2b
Given a meme (both the textual and the visual content), identify whether it contains a persuasion technique (at least one of the 22 techniques - https://propaganda.math.unipd.it/semeval2024task4/definitions22.html - we considered in this task), or no technique. This is a binary classification problem.

Note that **for all subtasks, there will be three surprise test datasets in different languages (a fourth one in English will be released as well), which will be revealed only at the final stages of the shared task. i.e. together with the release of the test data.**
This has the goal to test zero-shot approaches.
The hierarchy is basically a Directed Acyclic graph that groups subsets of the techniques that share similar characteristics in a hierarchical structure.

![hierarchy2](https://github.com/mharis-waqar/Multilingual-Detection-of-Persuasion-Techniques-in-Memes/assets/47879614/84bd0a0d-e3e0-4375-86be-62ae79db07f6)

## Input and Submission File Format

The input data for subtask 1 is the text extracted from the meme. The training, the development and the test sets for all subtasks are distributed as json files, one single file per subtask.
The input data for subtasks 2a and 2b, in addition to the text extracted from the meme, is the image of the meme itself. The images are distributed together with the subtask json in a zip file, and it is available, upon registration, from the personal page of your team.

Here is an example of a meme:

![125_image](https://github.com/mharis-waqar/Multilingual-Detection-of-Persuasion-Techniques-in-Memes/assets/47879614/0061ad4c-e3b2-44b4-a9db-15980315e863)

### Subtask 1

The entry for that example in the json file for subtask 1 is

		{
			"id": "125",
			"text": "I HATE TRUMP\n\nMOST TERRORIST DO",
			"labels": [
				"Loaded Language",
				"Name calling/Labeling"
		               ],
		        "link": "https://..."
		},		
		
where
id is the unique identifier of the example across all three subtasks
text is the textual content of the meme, as a single UTF-8 string. While the text is first extracted automatically from the meme, we manually post-process it to remove errors and to format it in such a way that each sentence is on a single row and blocks of text in different areas of the image are separated by a blank row. Note that task 1 is an NLP task since the image is not provided as an input.
labels is a list of valid technique names (the full list is available in your team page after registration) used in the text. Since these are the gold labels, they will be provided for the training set only. In this case two techniques were spotted: Loaded Language and Name calling/Labeling.
A submission for task 1 is a single json file with the same format as the input file, but where only the fields id, labels are required.
### Subtask 2a
The input for subtask 2a is a json and a folder with the images of the memes. The entry in the json file for the meme above is

		{
			"id": "125",
			"text": "I HATE TRUMP\n\nMOST TERRORIST DO",
			"labels": [
            				"Reductio ad hitlerum",
            				"Smears",
            				"Loaded Language",
            				"Name calling/Labeling"
        			],
            	        "image": "125_image.png",
			"link": "https://..."
		},		
		
where image is the name of the file with the image of the meme in the folder. The meaning of id, text and labels is the same as for task 1. However, the list of technique names is different (the full list is available in your team page after registration). Note that the field labels will be provided for the training set only, since it corresponds to the gold labels. Notice, however, that now we are able to see the image of the meme, hence we might be able to spot more techniques. In this example smears and Reductio ad hitlerum become evident only after we are able to understand who the two sentences are attributed to. There are other cases in which a technique is conveyed by the image only (see example with id 189 in the training set).
A submission for task 2 consists in a single json file with the same format as the input file, but where only the fields id, labels, for each example, are required.
### Subtask 2b
Subtask 2b is the same as subtask 2a. However, it is going to be evaluated as a binary task, whether at least one technique is present in the meme or no technique is present. Notice, these two labels correspond to the children of the root node of the hierarchy.
The entry for that example in the json file for subtask 1 is

		{
			"id": "125",
			"text": "I HATE TRUMP\n\nMOST TERRORIST DO",
			"label": "propagandistic"
		},		
		
## Code

Files for all the sub tasks are in Subtask (1, 2a, 2b).py present in the git repo.

### for SubTask 1:

We have fine tuned a Bert Based model on specific multilabel classification, by giving the annotation of train.json for training and validation.json for validations. And after that we performed testing on the provided dev-unlabeled dataset for multilabel classification.

### SubTask 2:
Not Done Yet

### Subtask 3:
Not Done Yet
