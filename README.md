# comp61332-coursework-1-question-classification-solved
**TO GET THIS SOLUTION VISIT:** [COMP61332 Coursework 1-Question classification Solved](https://www.ankitcodinghub.com/product/comp61332-coursework-1-question-classification-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;99623&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;COMP61332 Coursework 1-Question classification&nbsp;Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="section">
<div class="layoutArea">
<div class="column">
&nbsp;

1. Introduction

This coursework is a group project. Your task is to build two question classifiers using (i) bag-of-words and (ii) BiLSTM, which will have been covered in Weeks 3 and 4 of COMP61332, respectively.

Input:​a question (e.g., “How many points make up a perfect fivepin bowling score ?”)

Output:​ one of ​N predefined classes (e.g., NUM:count, which is the class label for questions that require counting, i.e., counting questions)

Intended Learning Outcomes

<ul>
<li>● &nbsp;to develop deep learning-based sentence classifiers using word embeddings and BiLSTM</li>
<li>● &nbsp;to evaluate and analyse your sentence classifiers according to different settings, such as
different word embeddings, and different parameters to fine-tune the models
</li>
<li>● &nbsp;to discuss your methods and results in the form of academic writing</li>
<li>● &nbsp;to act as a responsible member of a team, communicate with team mates, and contribute to
the team’s self-organisation, planning and conflict resolution for the duration of the group work

2. Instructions

Your implementation has to be in ​python3,​using PyTorch (​https://pytorch.org/​). If you are not familiar with PyTorch, check out some tutorials first (e.g., ​this article and its Pytorch tutorials 1, 2, and 3 at the bottom of the page, or this ​pytorch tutorial​).

2.1 Data
</li>
</ul>
In this coursework, we will make use of the dataset from https://cogcomp.seas.upenn.edu/Data/QA/QC/ (Training set 5, which contains 5500 labelled questions). Because there is no development set, you have to randomly split the training set into 10 portions. 9 portions are for training, and the other is for development, which will be used for early stopping or hyperparameter tuning. The test set will be the TREC 10 questions available from the same site.

2.2 Word embeddings

You are required to implement two kinds of word embeddings.

(1) ​Randomly initialised ​word embeddings.​ For example, to build a vocabulary, you can select those

words appearing at least ​k​ times in the training set.

(2) ​Pre-trained word embeddings​ such as word2vec (see this relevant ​video​) or GloVe.

Please note that your implementation should have an option to freeze or to fine-tune the pre-trained word embeddings during training. (​Although those options are applicable to randomly initialised word embeddings as well, you are not required to do so.​)

For preprocessing, you can ignore stop-words (e.g., “up”, “a”), or lowercase all words (e.g., “How” becomes “how”). Do not forget to handle words that are not in the vocabulary!

</div>
</div>
<div class="layoutArea">
<div class="column">
1

</div>
</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="section">
<div class="layoutArea">
<div class="column">
2.3 Sentence representations

</div>
</div>
<div class="layoutArea">
<div class="column">
2.3.1 Bag-of-words (BOW)

A bag-of-words is a set of words (we can ignore word frequency here). For instance, the bag-of-words of the question above is

bow(“How many points…”) =

{“How”, “many”, “points”, “make”, “up”, “a”, “perfect”, “fivepin”, “bowling”, “score”}

We can define a vector representing a bag-of-words as:

vec (s) = 1 ∑ vec(w) bow |bow(s)| w∈bow(s)

where ​s is a sentence/question and ​vec​bow(​ s) is the vector representation for ​s​. ​vec(w) is the vector representation for word ​w​.

For example:

vec(“How many points…”) =

1/10 * (vec(“How”) + vec(“many”) + … + vec(“score”))

2.3.2 BiLSTM

Please check ​this tutorial for using LSTM. You just need to do an extra step to replace LSTM by BiLSTM. Let’s denote the vector representation for a sentence ​s ​produced by BiLSTM as:

vecbilstm(s) = BiLSTM(s)

where ​BiLSTM(s) i​ s the vector returned by your implementation of BiLSTM.

2.4 Classifier

Given vec​bow​(s) or vec​bilstm(​ s) above, you will use a feed-forward neural network with a softmax output layer for classification. This feed-forward neural network for classification task is presented in ​this video​ in Week 3.

2.5 Classifier (plus)

You can build more sophisticated classifiers, by

<ul>
<li>● &nbsp;combining vec​bow(​ s) and vec​bilstm(​ s) into one vector vec(s), and/or</li>
<li>● &nbsp;combining several classifiers (i.e., ensemble).
2.6 Interface

Your main source code should be in a file named ​question_classifier.py

We should be able to run your code to train a model by issuing the following command:

<pre>% python3 question_classifier.py --train --config [configuration_file_path]
</pre>
We should also be able to run your code to test a model by issuing the following command:

<pre>% python3 question_classifier.py --test --config [configuration_file_path]
</pre>
The program will load a configuration file storing all needed information in different sections, such as:

​# Paths To Datasets And Evaluation path_train : ../data/train.txt path_dev : ../data/dev.txt path_test : ../data/test.txt
</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="section">
<div class="layoutArea">
<div class="column">
# Options for model

model : bow # bow, bilstm, bow_ensemble, bilstm_ensemble… path_model : ../data/model.bow

# Early Stopping early_stopping : 50

# Model Settings epoch : 10 lowercase : false

# Using pre-trained Embeddings path_pre_emb : ../data/glove.txt

# Network Structure word_embedding_dim : 200 batch_size : 20

# Hyperparameters lr_param : 0.0001

# Evaluation

path_eval_result : ../data/output.txt

# Ensemble model

model : bilstm_ensemble

ensemble_size : 5

path_model : ../data/model.bilstm_ensemble

Notes:​

– If your code supports more than two required models (as mentioned in Section 2.5), such as an

ensemble of 5 BiLSTM models, your configuration file may include the ‘​Ensemble model​’ section

as in the above example. In that case, the five BiLSTM models will be stored in

../data/model.bilstm_ensemble.0 ../data/model.bilstm_ensemble.1 … ../data/model.bilstm_ensemble.4

– Output (e.g., ​../data/output.txt​) is a file in which each line is a class for each testing question and the performance (i.e., accuracy).

– You may need to store some more information on the model (e.g., vocabulary). Do not hesitate to make use of the configuration file to specify paths to any information you might wish to store.

3. Deliverables

There are two deliverables for this coursework.

3.1 Your implementation (in a zip file)

<ul>
<li>● &nbsp;You can use any environment/operating system during development, but note that during marking, the code that you submit will be run on the virtual machine (VM) distributed by the Department of Computer Science.</li>
<li>● &nbsp;Only pytorch, numpy, and python3 standard libraries are allowed. You don’t need any off-the-shelf NLP libraries like NLTK, Spacy or/and sklearn for preprocessing the data. If you</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
3

</div>
</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="section">
<div class="layoutArea">
<div class="column">
want to remove stopwords, you can find a stopword list ​here​. Exceptionally, you can use

sklearn library for evaluation metrics, and other libraries for your interface. (Please check

section 5). In that case, you have to put required libraries into a file named

requirements.txt

● The implementation should come with three folders:

<ul>
<li>○ &nbsp;document​: a document containing a description for each function, a README file
with instructions on how to use the code
</li>
<li>○ &nbsp;data​: training, dev, test, configuration files and some extra files needed for your
models (e.g., vocabulary). For each model, you need one configuration file (e.g., bow.config, bilstm.config). Please note that you should not include your trained models or pre-trained word embeddings in the submission.
</li>
<li>○ &nbsp;src​: your source code.

3.2 A short paper reporting results

This should be in the form of a research paper (2-3 pages excluding references), such as https://2021.aclweb.org/downloads/acl-ijcnlp2021-templates.zip​. We highly recommend you to use latex with Overleaf, where you and your team can easily collaborate in writing https://www.overleaf.com/latex/templates/instructions-for-acl-ijcnlp-2021-proceedings/mhxffkjdwymb​. The report should contain at least the following points:

● Introduction
</li>
</ul>
<ul>
<li>○ &nbsp;What is the problem?</li>
<li>○ &nbsp;Why is it important to be able to classify questions?</li>
<li>○ &nbsp;Why is this task difficult or interesting?</li>
</ul>
<ul>
<li>● &nbsp;Describe your approaches, e.g.
<ul>
<li>○ &nbsp;How did you turn sentences into vectors?</li>
<li>○ &nbsp;What are your models?</li>
</ul>
</li>
<li>● &nbsp;Describe your experiments, e.g.
<ul>
<li>○ &nbsp;Experiment set-up,
<ul>
<li>What is the data split that you used?</li>
<li>Describe your preprocessing steps (e.g. removing stopwords, lowercasing
words.)
</li>
<li>Specify the performance metrics that you used. You don’t need to explain
them in detail since those details will be covered in Week 5.
</li>
</ul>
</li>
<li>○ &nbsp;Results: There are 8 possible combinations of the results. ​In the paper, you should
report at least 6 sets​: 2 models x 3 word embedding settings including random inititialisation without fine-tuning, pretrained embeddings with and without fine-tuning.
</li>
<li>○ &nbsp;Ablation study, e.g.
<ul>
<li>What happens if you freeze/fine-tune the pre-trained word embeddings?</li>
<li>What happens if you use randomly initialized word embeddings instead of
pre-trained word embeddings?
</li>
</ul>
</li>
<li>○ &nbsp;Some in-depth analyses, e.g.
<ul>
<li>What happens if you use only part of the training set?</li>
<li>Which classes are more difficult to classify?</li>
<li>Confusion matrix?</li>
<li>What is the effect of using other preprocessing steps?
4. Marking scheme

This coursework accounts for 25% of your final mark for COMP61332, and is worth 100 points. The following rubric will be used in marking your group project, where the first column specifies the various criteria and the second column indicates the maximum number marks your group can be possibly given.
</li>
</ul>
</li>
</ul>
</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
4

</div>
</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="section">
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Source code organisation

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Interface

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Model implementation

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Training BOW model

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Training BiLSTM model

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Word embedding options

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Freeze/fine-tune pre-trained embeddings

</div>
</div>
</td>
</tr>
</tbody>
</table>
<div class="layoutArea">
<div class="column">
Bonus

</div>
<div class="column">
Implementation of Question Classifiers

0 The code does not have three subfolders as required, no documentation, no readme file about how to run the code.

2 The code only has one of the following things: (1) subfolders, (2) documentation, (3) readme file with ​requirements.txt ​(if any)

5 The code is well structured and documented as required

0 The code cannot run by using the required command line

2 The code can run by using the required command line

5 The code allows us to input different options via configuration files 0 No model is implemented

5 Only BOW model is implemented and run without errors

10 Two models (BOW and BiLSTM) are implemented and run without errors

0 After 10 epochs, the accuracy is less than 40%

5 After 10 epochs, the accuracy is from 40%-50%

10 After 10 epochs, the accuracy is more than 50%

0 After 10 epochs, the accuracy is less than 40%

5 After 10 epochs, the accuracy is from 40%-50%

10 After 10 epochs, the accuracy is more than 50%

0 The yielded training losses and accuracies are not different with different options

5 The yielded training losses and accuracies are different with different options

0 The yielded training losses and accuracies are not different with different options

5 The yielded training losses and accuracies are different with different options

10 Extra model(s) is implemented and run properly

Short Paper

0 The short paper looks like a technical report/user manual than a research paper

5

10

</div>
</div>
<div class="layoutArea">
<div class="column">
Academic writing

</div>
</div>
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
The short paper was written for an academic audience. However there are some points that were not clearly presented, or it seemed like the discussion lacks originality/argumentation.

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
The short paper was written for an academic audience and can potentially be published in a research workshop or symposium. Ideas were presented in a clear and well-argued manner.

</div>
</div>
</td>
</tr>
</tbody>
</table>
<div class="layoutArea">
<div class="column">
5

</div>
</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="section">
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Background and introduction

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Methodology

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Experiments

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Analysis and interpretation

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Exemplification/v isualisation

</div>
</div>
</td>
</tr>
</tbody>
</table>
<div class="layoutArea">
<div class="column">
0

</div>
</div>
<div class="layoutArea">
<div class="column">
The paper does not provide an introduction to the task of question classification (e.g., why it is interesting) and does not clearly present the research questions that the project seeks to answer.

</div>
</div>
<div class="layoutArea">
<div class="column">
5 The paper provides an introduction to the proposed topic and presents the analytical questions that the project seeks to answer.

0 The paper does not provide sufficient details on the group’s methodology.

5

10

0 The paper does not provide any details about experiments, including experimental data, preprocessing steps, and the evaluation metrics.

2

5

0

5

10

0 The group did not provide any examples nor make use of visualisation to evidence their analysis and interpretation of quantitative results.

5

10 The group explained their findings and interpretation by providing supporting examples or making use of suitable visualisation.

Your deliverables will be assessed based on the marking scheme above, which will lead to one overall mark (out of 100). Everyone in your group will get the same mark: one of the Learning Outcomes of this coursework is focussed on acting as a responsible team member (see Intended Learning Outcomes in Section 1), hence it is everyone’s duty to ensure that tasks are delegated fairly, that there are equal contributions, and that integration goes smoothly.

In the exceptional case where one or more group members have not put in an acceptable contribution, despite the team’s effort to bring them back into the team and suitable discussions of

</div>
</div>
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
The paper provides details on the group’s methodology, including how to turn sentences into vectors and description about the BOW and BiLSTM models. However some parts need further elaboration.

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
The paper provides sufficient and clear details on the group’s methodology, including how to turn sentences into vectors and description about the BOW and BiLSTM models.

</div>
</div>
</td>
</tr>
</tbody>
</table>
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
The paper provides details about experiments, including experimental data, preprocessing steps, and the evaluation metrics. However, some parts need further elaboration.

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
The paper provides sufficient and clear details about experiments, including experimental data, preprocessing steps, and the evaluation metrics.

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Quantitative results were obtained by the implemented classifiers but were not analysed and interpreted in order to answer the research questions set out in the background and introduction section.

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Quantitative results obtained by the implemented classifiers were analysed in order to answer the research questions set out in the background and introduction section, but in some parts the interpretation seems exaggerated (or are not aligned with the results).

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Quantitative results obtained by the implemented classifiers were adequately interpreted, allowing the group to answer the research questions they set out in the background and introduction section.

</div>
</div>
</td>
</tr>
</tbody>
</table>
<div class="layoutArea">
<div class="column">
The group provided examples and made use of visualisation, however some of these do not support or are not aligned with their findings/interpretation.

</div>
</div>
<div class="layoutArea">
<div class="column">
6

</div>
</div>
</div>
</div>
<div class="page" title="Page 7">
<div class="section">
<div class="layoutArea">
<div class="column">
the issues arising, we implement a grievance procedure. A group can bring forward a “case of grievance” to the COMP61332 teaching staff by providing the following pieces of evidence:

<ul>
<li>● &nbsp;minutes of team meetings</li>
<li>● &nbsp;a written description of the events that led to the break-down of the team work, including a
description of the actions that were taken to get the team back on track.

The case should be brought forward to the teaching staff no later than one working week after the coursework deadline. The COMP61332 teaching staff will then decide whether the situation is indeed exceptional and warrants a mark re-distribution.

5. Suggestions

5.1 Embeddings

For randomly initialised word embeddings, use the class

https://pytorch.org/docs/stable/nn.html?highlight=embedding#torch.nn.Embedding

For using pre-trained word embeddings, use the function

https://pytorch.org/docs/stable/nn.html?highlight=embedding#torch.nn.Embedding.from_pretrained

5.1.1 Handling unknown words

To handle unknown words, you can use ​#UNK# tokens. The embedding of ​#UNK# can be initialised randomly by drawing from N(0, 1).

5.1.2 Pruning pre-trained embeddings

Instead of loading the whole pre-trained word embeddings, you can remove the embeddings of those words that do not appear in the dataset. You can download glove.small.zip from the course page on Blackboard. The word embeddings file also includes an embedding for ​#UNK#​.

5.1.3 Fix random seeds

To make sure the three options of word embeddings are properly implemented, we should fix random seeds by adding the following lines to the beginning of ​question_classifiation.py

import torch, random torch.manual_seed(1) random.seed(1)

5.3 BiLSTM using Pytorch

As aforementioned, please check ​this tutorial for using LSTM in Pytorch first. For BiLSTM, you just need to set the ​bidirectional parameter of LSTM to true, as explained ​here​. For example, a tagger using BiLSTM can be coded as follows:
</li>
</ul>
</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
import​ torch

import​ torch.nn ​as​ nn

class​ ​BiLSTMTagger​(​nn​.​Module​):

​def​ ​__init__​(​self​, ​embedding_dim​, ​hidden_dim​, ​vocab_size​, ​tagset_size​): ​super​(LSTMTagger, ​self​).​__init__​()

​self​.hidden_dim = hidden_dim

​self​.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
7

</div>
</div>
</div>
</div>
<div class="page" title="Page 8">
<div class="section">
<div class="section">
<div class="layoutArea">
<div class="column">
​# The BiLSTM takes word embeddings as inputs, and outputs hidden states ​# with dimensionality hidden_dim.

​self​.lstm = nn.LSTM(embedding_dim, hidden_dim, ​bidirectional​=​True​)

​# The linear layer that maps from hidden state space to tag space ​self​.hidden2tag = nn.Linear(hidden_dim, tagset_size)

</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
5.4 Evaluation metrics

For this classification task, you can use both accuracy and F1 scores. Please check this python library for more information: ​https://scikit-learn.org/stable/modules/model_evaluation.html.​ As aforementioned, you don’t need to explain them in detail in your report; those details will be covered in Week 5 anyway.

5.5 Reading configuration files

There are several ways to store configuration files. One way is to store configuration information in the same structure as Windows INI files and use ​configparser​, a python library to read the file.

For example, given this config.ini file:

We can parse it as follows:

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
[PATH]

path_train​ = ../data/train.txt path_dev​ = ../data/dev.txt path_test​ = ../data/test.txt

</div>
</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
import​ configparser

config = configparser.ConfigParser() config.sections()

config.read(​”config.ini”​)

print​(config.keys())

path_train = config[​”PATH”​][​”path_train”​] path_dev = config[​”PATH”​][​”path_dev”​] path_test = config[​”PATH”​][​”path_test”​]

</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
Another option is to use pyyaml: ​https://pyyaml.org/wiki/PyYAMLDocumentation 5.6 Parsing command line arguments

To parse command line arguments, we can use ​argparse​, a standard Python library. Following is an example of parsing arguments so that you can run your code as described in Section 2.6:

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
import​ argparse

parser = argparse.ArgumentParser()

parser.add_argument(​’–config’​, ​type​=​str​, ​required​=​True​, ​help​=​’Configuration file’​)

parser.add_argument(​’–train’​, ​action​=​’store_true’​, ​help​=​’Training mode – model is saved’​)

</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
8

</div>
</div>
</div>
</div>
<div class="page" title="Page 9">
<div class="section">
<div class="section">
<div class="layoutArea">
<div class="column">
parser.add_argument(​’–test’​, ​action​=​’store_true’​, ​help​=​’Testing mode – needs a model to load’​)

args = parser.parse_args()

if​ args.train:

​#call train function

train(args.config) elif​ args.test:

​#call test function test(args.config)

</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
Deadline: 12:00AM (Midnight, UK time), Friday, 12th March, via Blackboard. Good luck!

</div>
</div>
<div class="layoutArea">
<div class="column">
9

</div>
</div>
</div>
</div>
