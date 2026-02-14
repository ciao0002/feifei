

# CoSLight: Co-optimizing Collaborator Selection and Decision-making to Enhance Traffic Signal Control



## ABSTRACT

Effective multi-intersection collaboration is pivotal for reinforcement-learning-based traffic signal control to alleviate congestion. Existing work mainly chooses neighboring intersections as collaborators. However, quite a lot of congestion, even some wide-range congestion, is caused by non-neighbors failing to collaborate. To address these issues, we propose to separate the collaborator selection as a second policy to be learned, concurrently being updated with the original signal-controlling policy. Specifically, the selection policy in real-time adaptively selects the best teammates according to phase- and intersection-level features. Empirical results on both synthetic and real-world datasets provide robust validation for the superiority of our approach, offering significant improvements over existing state-of-the-art methods. Code is available at https://github.com/bonaldli/CoSLight.

### CCS CONCEPTS

• Computing methodologies → Neural networks; Partially-observable Markov decision processes; Multi-agent reinforcement learning; Sequential decision making.

### KEYWORDS

Traffic Signal Control, Multi-intersection Transportation, Multi-agent Systems, Multi-agent Reinforcement Learning

* The corresponding author.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//58d928b7-5645-42cf-9dd1-a2060dc94664/markdown_1/imgs/img_in_image_box_106_1287_238_1336.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A17Z%2F-1%2F%2F5c5f3a63adb3fbaf41ac0e9bea9e5c7e96ed6a5316122f8bab8a241c21922fe7" alt="Image" width="10%" /></div>


This work is licensed under a Creative Commons Attribution International 4.0 License.

Rui Zhao

rzhao@ee.cuhk.edu.hk

Qing Yuan Research Institute of Shanghai Jiao Tong University Shanghai, China

##### ACM Reference Format:

KDD '24, August 25–29, 2024, Barcelona, Spain  

© 2024 Copyright held by the owner/author(s).  

ACM ISBN 979-8-4007-0490-1/24/08  

https://doi.org/10.1145/3637528.3671998

Jingqing Ruan, Ziyue Li*, Hua Wei, Haoyuan Jiang, Jiaming Lu, Xuantang Xiong, Hangyu Mao*, and Rui Zhao. 2024. CoSLight: Co-optimizing Collaborator Selection and Decision-making to Enhance Traffic Signal Control. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '24), August 25–29, 2024, Barcelona, Spain. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3637528.3671998

## 1 INTRODUCTION

Traffic Signal Control (TSC) plays a critical role in managing urban traffic flow and alleviating congestion. Over the past decade, multi-agent reinforcement learning (MARL) has emerged as a powerful tool for optimizing TSC [11, 15, 30, 53]. However, effectively promoting multi-intersection collaboration remains a persistent challenge in applying MARL to TSC.

Traditionally, researchers have treated geographically adjacent intersections as natural collaborators, and combined MARL methods with graph neural networks (GNNs) [4, 9, 24, 34, 52, 56, 60] and approaches multi-level embeddings [14, 17, 22, 32, 36, 64] to model multi-intersection collaboration. However, the collaboration among intersections might be far beyond topological proximity in the real world [57, 67]. For instance, during morning rush hours, signals from residential to business areas must be strategically coordinated to facilitate driving in town. As shown in Figure 1(a), intersections in upper streams should regulate incoming traffic to prevent downstream congestion, requiring them to synchronize with signals closer to business districts, which direct traffic towards parking and alternate routes. This coordination goes beyond mere proximity, emphasizing the need for dynamic and non-adjacent collaboration across the network. Likewise, during evening rush hours in Figure 1(b), signals from business to residential areas also need to cooperate for better driving out of town. The coordination between areas of intersections in the morning and evening peaks

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//58d928b7-5645-42cf-9dd1-a2060dc94664/markdown_2/imgs/img_in_image_box_146_165_315_305.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A18Z%2F-1%2F%2F28985835d1ca44c227956b8980c7303e04f11dbeb0b32683189c001fbbcb5d73" alt="Image" width="13%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//58d928b7-5645-42cf-9dd1-a2060dc94664/markdown_2/imgs/img_in_image_box_316_167_551_306.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A18Z%2F-1%2F%2Fb58204339b447b45ec4b5ca0190a5453941d6c27cd28ab84de012269c510a18d" alt="Image" width="19%" /></div>


<div style="text-align: center;">(a) Morning Rush Hours</div>


<div style="text-align: center;">(b) Evening Rush Hours</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//58d928b7-5645-42cf-9dd1-a2060dc94664/markdown_2/imgs/img_in_chart_box_128_351_321_487.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A18Z%2F-1%2F%2F05de6214eab0897cec5d1ee57ddc967f681c3bf81410e976b78cb598e836b934" alt="Image" width="15%" /></div>


<div style="text-align: center;">(c) Fixed  $ \rho $, with Optimized  $ \theta $</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//58d928b7-5645-42cf-9dd1-a2060dc94664/markdown_2/imgs/img_in_chart_box_350_352_556_486.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A18Z%2F-1%2F%2F124d9743ec67e62b55bda7bf1e832864a86f9b090f1879686a0d83b4c7af66ba" alt="Image" width="16%" /></div>


<div style="text-align: center;">(d) Joint Optimization of  $ \rho $ and  $ \theta $</div>


<div style="text-align: center;">Figure 1: (a)-(b): The coordination between areas of intersections during rush hours; (c)-(d): The collaboration policy  $ \rho $ and decision policy  $ \theta $ should be jointly optimized to prevent suboptimal.</div>


mostly differ due to different origin-destination flows and other factors such as weather and trip purpose.

In this paper, we propose CoSLight to investigate the collaboration among intersections beyond topological neighbors, which comes along with two questions:

• Whom to collaborate for better decisions? While existing methods relying on heavy GNNs to learn whom to collaborate require information propagation within the whole network, in real-time, the collaborator selection must be light and agile. In this paper, we utilize a simple two-layer MLP structure for top-k collaborator selection, which not only reduces computational complexity [13, 40, 49] but also achieves better performance in experiments. Instead of counting on GNNs to learn the relationships by themselves, we incorporate two golden rules in the MLP collaboration matrix: that "you are your biggest collaborator" and "mutual reciprocity", which penalize the diagonal to be the largest and the matrix to be symmetric. These rules significantly improve collaboration benefits.

• How to collaborate and optimize decisions? The goodness of collaborator selection largely influences the goodness of the decision policy. For example, the traditional practice is to first select collaborators without training and then to decide the signal policy, as shown in Figure 1(c), which comes with two drawbacks: (1) the decision policy for one intersection might need a longer time to adjust to its collaborators, (2) the decision policy is optimized towards maximizing the cumulative reward, while the collaborators are selected separately without acknowledgment of the performance of decision policy. To address this challenge, in this paper, we design a joint optimization scheme that simultaneously trains the collaboration selection policy and decision policy to maximize the cumulative return through a joint policy gradient, reinforcing the strong coupling between collaboration policy and decision policy.

We conduct comprehensive experiments using both synthetic-and real data with different traffic flow and network structures. Our method consistently outperforms state-of-the-art RL methods, which shows that the effectiveness of collaborator selection and joint optimization with decision policy. We further showcase that the selected collaborators are not necessarily geographic neighbors, and visualize several interesting collaboration strategies learned by our method to show that our collaborator selection is effective and generalizable to different road networks.

In summary, our contribution is the following:

- CoSLight is the first work to decouple and co-optimize the collaborator selection policy and signal planning policy.

- Specifically, CoSLight combines a Dual-Feature Extractor, capturing both phase- and intersection-level features, with a Multi-Intersection Collaboration module designed to strategically choose cooperating intersections. Moreover, a joint optimization strategy is derived to co-train the collaborator selection and decision-making policies, which uses a joint policy gradient to enhance the cumulative return, emphasizing the interdependence between collaboration selection and decision-making.

• Extensive evaluations and multi-dimensional analysis using synthetic and real-world datasets demonstrate the effectiveness and superiority of CoSLight.

## 2 RELATED WORK

In this section, we review existing approaches based on how they collaborate: implicit and explicit collaboration.

Implicit collaboration only accesses information from other agents during the update phase to assist gradient backpropagation. MPLight repurposes FRAP [45, 66] for a multi-agent context. IDQN [7, 8, 10, 31], IPPO [1] and MAPPO [59] tackle TSC problems directly from the MARL perspective. Works like [20, 42, 50] extend the single-agent DQN solution to multi-agent scenarios using the max-plus coordination algorithm. Other methods such as IntelliLight [54] and PressLight [51] enrich the state space by leveraging different types of additional information, like image frames or max pressure, respectively. FMA2C [26] and FedLight [58] consider collaborative optimization from federated RL. However, these methods focus on multi-intersection information from an update mechanism or gradient design perspective, while the decentralized execution leads to weak attention and may hinder efficient multi-intersection collaboration.

Explicit collaboration allows accessing other agents' information during the decision-making process to enhance collaboration. To overcome the shortcomings of weak collaboration in implicit strategies, explicit collaboration methods can be further categorized into two sub-classes. One line of research focuses on multi-intersection representation extraction, such as CoLight [30, 52], DynSTGAT [56], IG-RL [9], MaCAR [60], and MetaGAT [24], which utilize GNNs as the feature extractor to model representations of the current intersection and its neighbors. X-Light [16] instead feeds the neighbors' MDP information into a Transformer. However, these methods risk introducing noise, such as unrelated intersection features, into collaboration; it also suffers from the computational complexity of matrix multiplication. Another line of research leverages group-based cooperation [27, 35, 36, 47, 48]. MT-GAD [18], and JointLight [21] uses heuristic grouping, which requires manual design, while CGB-MATSC [46] applies KNN grouping directly based on state observation, a non-parametric method. GeneraLight [62]

utilizes k-means to execute the flow clustering to learn diverse meta-parameters. GPLight [23] utilizes mutual information and gathering constraints to derive latent representations, which are subsequently clustered into groups. However, their grouping strategies cannot be directly co-optimized with the decision-making process using the reward signal in RL, leading to a potentially suboptimal signal policy.

In contrast to these existing methods, we introduce a dual-feature extractor to derive collaborator representations beneficial for collaboration. Furthermore, we propose CoS policy co-learned with decision policy with MIRL. This allows for the selection of optimal collaborators and primitive action for each intersection, guiding intense collaboration among multiple intersections.

## 3 PRELIMINARY AND PROBLEM STATEMENT

### 3.1 Preliminary

Firstly, we introduce some fundamental concepts related to traffic signal control, including traffic movement, signal phase, traffic intersection, multi-intersection traffic signal control, queue length, and pressure. In Figure 2, we give a visual representation of these concepts to further aid understanding.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//58d928b7-5645-42cf-9dd1-a2060dc94664/markdown_3/imgs/img_in_image_box_107_675_584_862.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A19Z%2F-1%2F%2Fecfad3c707bbde16ff301f49c5b10b002e3cbaf62cd558f309d8f3bfdbc2d8a0" alt="Image" width="38%" /></div>


<div style="text-align: center;">Figure 2: (a) The illustration of intersection. (b) There are 12 movements: [North, South, West, East (four approaches)] × [Left, Go-through, Right (three directions)]. Usually, turning right isn’t signal-controlled, so only 8 movements (index from 1-8 in (a)) are signal-controlled. (c) A phase is two non-conflicting movements that can be released together. There are 8 phases, e.g., phase-A combines movements 1 and 5. (d) The signal-control policy is to select one phase for the next time step according to the traffic condition.</div>


Traffic Intersection. A traffic intersection, where multiple roads intersect, uses traffic signals to control vehicle flow. Figure 2 depicts a four-arm intersection, equipped with lanes designated for left turns, straight travel, and right turns. It features four roads and twelve lanes for both entering and exiting traffic. We denote the incoming and outgoing lanes at intersection i as  $ \mathcal{L}_{in}^{i} $ and  $ \mathcal{L}_{out}^{i} $.

Traffic Movement. It refers to the progression of vehicles across an intersection in a specific direction, namely, turning left, going straight, or turning right. As depicted in Figure 2, twelve distinct traffic movements can be identified. Right-turning vehicles usually proceed regardless of the signal status.

Signal Phase. It is a set of two traffic movements that can be released together without conflicts. As illustrated in Figure 2, the intersection comprises eight distinct phases A-H.

Multi-Intersection Traffic Signal Control (TSC). At each intersection, an RL agent is deployed to manage the traffic signal control. During each time unit denoted as  $ t_{duration} $, the RL agent i observes the state of the environment, represented as  $ o_{t}^{i} $. It then determines the action  $ a_{t}^{i} $, which dictates the signal phase for intersection i. The objective of the agent i is to choose an optimal action (i.e., determining the most appropriate signal phase) to maximize the cumulative reward.

Queue Length. The queue length at each intersection i is defined as the aggregate length of vehicle queues in all incoming lanes toward the intersection, denoted as:

 $$ Q_{l e n}^{i}=\sum q(l),l\in\mathcal{L}_{i n}^{i}, $$ 

where  $ q(l) $ is the queue length in the lane l.

Pressure. Pressure can be divided into intersection-wise and phase-wise categories. Intersection-wise pressure measures the imbalance between incoming and outgoing vehicle queues at an intersection, indicating traffic load discrepancies. Phase-wise pressure concerns a specific signal phase  $ p $. Each signal phase permits several traffic movements, each marked by  $ (l,m) $. Let  $ x(l,m) $ signify the vehicle count difference between lane  $ l $ and lane  $ m $ for a movement  $ (l,m) $, and the phase-wise pressure  $ P(p) $ denotes the cumulative sum of the pressures of all permissible movements within that phase, i.e.,  $ \sum_{(l,m)} x(l,m) $.

Collaborator Matters. As shown in Figure 3, we conducted experiments on Grid 4 × 4 and Grid 5 × 5 to further substantiate our assertion: the optimal number and range of collaborating intersections vary across different scenarios. For instance, in Grid 4 × 4, the impact of collaboration remains largely consistent regardless of increasing distances, suggesting that the benefits of collaboration might be distance-independent in this setting. Alternatively, it could indicate that merely selecting topologically adjacent intersections might not enhance the collaborative outcomes. In contrast, Grid 5 × 5 displays a negative impact with one-hop collaboration, whereas two-hop collaboration produces the greatest benefits. This underscores the significance of precisely and judiciously selecting collaborators, highlighting that not just the presence, but the quality and context of collaboration matters.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//58d928b7-5645-42cf-9dd1-a2060dc94664/markdown_3/imgs/img_in_chart_box_666_1041_870_1214.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A19Z%2F-1%2F%2Fe30ceae2ab4ca17cee713543915d75799b76cb51e8253ad57408b1381238cd89" alt="Image" width="16%" /></div>


<div style="text-align: center;">(a) Grid  $ 4 \times 4 $</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//58d928b7-5645-42cf-9dd1-a2060dc94664/markdown_3/imgs/img_in_chart_box_881_1043_1084_1213.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A19Z%2F-1%2F%2F7d30db836358519bf74a21824cb4ff9903ccb1ac2d6e791e2515ca2729b25b65" alt="Image" width="16%" /></div>


<div style="text-align: center;">(b) Grid  $ 5 \times 5 $</div>


<div style="text-align: center;">Figure 3: Gains over hops: choosing the right collaborators is important.</div>


### 3.2 Problem Statement

Then, we formulate  $ N $-intersection problem based on multiagent Markov decision processes (MMDPs) [5] as collaborator-based MMDPs, which can be expressed as  $ \langle I\rangle, S, O, \{C^i\}_{i=1}^N, \{\mathcal{A}^i\}_{i=1}^N $,

 $ \mathbb{P}, r, \gamma > i \in I $ is the  $ i^th $ intersection,  $ N $ is the number of intersections, and  $ \mathcal{S}, O $ are the global state space and local observation space.  $ C^i $ and  $ \mathcal{A}^i $ denote the selected collaborator and action space for the  $ i^th $ intersection. We label  $ \text{ids} := (\text{ids}^1, ..., \text{ids}^N) \in C $ and  $ \boldsymbol{a} := (a^1, ..., a^N) \in \mathcal{A} $ the joint collaborator identifiers and actions for all intersections.  $ \mathbb{P}(\cdot | \text{s}, \text{ids}, \boldsymbol{a}) $ is the transition dynamics. All intersections share the same reward function  $ r(s, \text{ids}, \boldsymbol{a}) $:  $ \mathcal{S} \times \mathcal{C} \times \mathcal{A} \to \mathbb{R} $.  $ \gamma \in (0, 1) $ denotes a discount factor. Here, we can denote  $ \tau = (s_0, \text{id}_s_0, a_0, s_1, \ldots) $ as the trajectory induced by the policy  $ \pi^{all} = \{\rho^i \cdot \pi^i\}_{i=1}^N\} $, where  $ \rho^i $ denotes the collaborator selection policy, and  $ \pi^i $ is the decision policy. All the intersections coordinate together to maximize the cumulative discounted return  $ \mathbb{E}_{\boldsymbol{\tau} \sim \pi^{all}} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, \text{id}_s_t, a_t) \right] $. We can define the overall joint policy as the product of selection policy  $ \rho $ and decision policy  $ \pi $ based on Bayesian chain rules.

 $$ \pi^{all}(ids,a|s)=\prod_{i=1}^{N}\rho^{i}(ids^{i}|o^{i})\times\pi^{i}(a^{i}|o^{i},ids^{i}). $$ 

## 4 METHODOLOGY

### 4.1 Main Modules

As shown in Figure 4, the dual-feature extractor pre-processes each intersection by obtaining phase- and intersection-level representations. Subsequently, the CoS identifies the most appropriate collaborators and collects their information to assist with the decision-making process.

4.1.1 Dual-Feature Extractor. To optimize policy-making in TSC, a comprehensive representation of traffic situations is vital. While phase-level features inform individual intersections, they lack broader coordination insights from correlated intersections. Thus, we complement with intersection-level features to better capture network-wide traffic dynamics based on Transformer [44].

At the phase level, we adopt FRAP [66] to obtain the phase-level representation. The raw observations o from the simulator include K features, such as the number of vehicles, queue length, the current phase, the flow, etc. For any traffic movement  $ m \in \{1, \ldots, 8\} $ in an intersection i, the k-th feature in the raw observation can be denoted as  $ o_m, k^i $. For brevity, the superscript i is omitted hereinafter. The process can be summarized as follows:  $ \mathbf{e}_m = \|k_m - \sigma(\text{MLP}_k(o_m, k)) $,  $ \mathbf{e}_{pcr} = \text{FRAP}(\mathbf{e}_{m_1}, \ldots, \mathbf{e}_{m_9}) $,  $ \mathbf{e}_{pr} = \text{MLP}(\mathbf{e}_{pcr}) $, where  $ || $ is concatenation,  $ \sigma $ is the activation function, and  $ \mathbf{e}_m $ is the embedding of traffic movement. Then FRAP( $ \cdot $) module (details in Appendix B) is applied to extract the phase competition representation  $ e_{pcr} $, and we reshape  $ e_{pcr} $ as a vector through flatten operation. Finally,  $ \mathbf{e}_{pr} $ is the phase representation in the intersection i.

At the intersection level, we adopt the Transformer encoder [28, 29, 61] as the backbone to model the relationship across multiple intersections: it takes in the phase representations  $ \boldsymbol{e}_{pr} = \{\boldsymbol{e}_{pr}^{i}\}_{i=1}^{N}, \boldsymbol{e}_{pr} \in \mathbb{R}^{N \times d} $,  $ d $ is the input dimension. Attention can be calculated to obtain the intersection-level representation  $ \boldsymbol{e}_{ir} $, attending to information from different intersections' representations.

 $$ \begin{array}{r}{\boldsymbol{e}_{i r}=\mathrm{T r a s f o r m e r E n c}(\boldsymbol{e}_{p r}),\quad\boldsymbol{e}_{i r}\in\mathbb{R}^{N\times d},}\end{array} $$ 

The TransformerEnc is the standard Transformer encoder with multi-head, followed by a feed-forward network.

In summary, by extracting features from both the phase- and intersection-level simultaneously, we are able to obtain a more comprehensive insight to provide sufficient representation for subsequent modeling.

4.1.2 Multi-Intersection Collaboration. The module facilitates cooperation among multiple intersections. By leveraging the rich representation  $ e_{ir} $, we design the CoS to select suitable collaborators for each intersection.

As mentioned before, since the collaboration policy (CoS) will be co-optimized with the decision policy, a lite and fast module is preferred to avoid becoming the computation bottleneck. Thus, we pursue minimalism: to use two-stacked MLPs. In computer vision, MLP-Mixer [40] is exclusively based on MLP and attains competitive scores as CNN-based methods or even ViT. In the spatiotemporal domain, MLPInit [13] also proves that MLPs could be 600× faster than GNN and achieves comparable results since MLP is free from matrix multiplication. GFS [49] gives an insightful and theoretically-rigid explanation: MLPs with general layer normalization provide similar functionality and effectiveness of aggregating nodes. Moreover, an ablation study by replacing MLPs with GNNs in Section 5.5 empirically confirms the effectiveness of MLPs.

Thus, the top-k Collaborator Selection (CoS) policy is implemented by two-stacked MLP layers,  $ \boldsymbol{\alpha}^i = f_{CoS}(\boldsymbol{e}_{ir}) = \mathrm{MLP}(\boldsymbol{e}_{ir}) $, where  $ \boldsymbol{\alpha}^i $ represents the logits from the MLP. Then the probability distribution can be constructed:

 $$ \mathcal{P}^{i}:=\mathrm{C a t e g o r i c a l}\left(\frac{\exp(\alpha_{j}^{i})}{\sum_{j^{\prime}\in\mathcal{I}}\exp(\alpha_{j^{\prime}}^{i})},j\in\mathcal{I}\right), $$ 

from which the indices of the selected collaborators for the intersection i can be sampled. Here, we set the hyper-parameter k as 5. We can sample the top-k collaborator identifiers without replacement from  $ \mathcal{P}^{i} $, which means sampling the first element, then renormalizing the remaining probabilities to sample the next element. Let  $ ID_{1}^{*}, \ldots, ID_{k}^{*} $ be an (ordered) sample without replacement from the  $ \mathcal{P}^{i} $, then the joint probability can be formulated as follows.

 $$ P(ID_{1}^{*}=id_{1}^{*},...,ID_{k}^{*}=id_{K}^{*})=\prod_{i=1}^{k}\frac{\exp(\alpha_{id_{i}^{*}}^{i})}{\sum_{i\in T_{i}^{*}}\exp(\alpha_{i}^{i})}, $$ 

where  $ id_1^*, \ldots, id_k^* = \arg \text{top-} k(\mathcal{P}^i) $ is the sampled top-K realization of variables  $ ID_1^*, \ldots, ID_k^* $, and  $ I_i^* = I \setminus \{id_1^*, \ldots, id_{i-1}^*\} $ is the domain (without replacement) for the  $ i $-th sampled element. Summarized as  $ ids^i \sim \rho^i(\mathbf{e}_{ir}) $, where  $ ids^i \in \mathbb{Z}^{1 \times k} $ refers to the indices of the selected collaborators for intersection  $ i $, and  $ \rho $ represents the CoS, whose parameters are shared among all intersections.

With $ids^i$ and $e_{ir}$, we obtain the $k$ collaborators' representations:

$e_{team}^i = \text{lookup}(e_{ir}, \text{ids}^i)$, where lookup operator extracts the vector at index $ids^i \in \mathbb{Z}^{1 \times k}$ from matrix $e_{ir} \in \mathbb{R}^{N \times d}$, thus $e_{team}^i \in \mathbb{R}^{k \times d}$. After that, mean-pooling is used to obtain the final collaboration representation. $\bar{e}_{team}^i = \text{mean-pooling}(e_{team}^i)$, $\bar{e}_{team}^i \in \mathbb{R}^{1 \times d}$.

For the decision policy  $ \pi^i $, it receives the intersection-level representation and collaborator representation and outputs the action  $ a^i $ for the intersection  $ i: a^i = \pi^i(\cdot | [e_{ir}^i || \tilde{e}_{team}^i]) $.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_0/imgs/img_in_image_box_155_178_1069_570.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A23Z%2F-1%2F%2F111adcc5a4659988f86779bde99d0251f1ab5c1c3c0163fced68494aa98ddca5" alt="Image" width="74%" /></div>


<div style="text-align: center;">Figure 4: Overview of our proposed CoSLight: integrating Dual-Feature Extractor for phase- and intersection-level features with the module of Multi-Intersection Collaboration to select teammates for cooperation.</div>


### 4.2 Optimization Scheme

We have introduced the inference process of the main modules. In this section, we propose an end-to-end joint optimization scheme to obtain the optimal CoS  $ \rho $ and decision policy  $ \pi $.

4.2.1 Overall Optimization Objective. The overall objective  $ \eta $ is to maximize the cumulative return, formulated as follows:

 $$ \max_{\varphi,\theta}\mathcal{J}(\varphi,\theta)=\max_{\varphi,\theta}\mathbb{E}_{ids\sim\rho_{\varphi}\atop a\sim\pi_{\theta}(\cdot|s,ids)}\left[\sum_{t=0}^{\infty}\gamma^{t}r(s_{t},a_{t})\right], $$ 

where  $ \rho = \{\rho^1, ..., \rho^N\} $ and  $ \pi = \{\pi^1, ..., \pi^N\} $ are joint CoS policy and decision policy, parameterized by  $ \{\varphi^1, ..., \varphi^N\} $ and  $ \{\theta^1, ..., \theta^N\} $, respectively.

According to Bellman Equation [3, 39], the Q-function provides an estimate to guide the agent toward the optimal policy that maximizes cumulative rewards. Thus, the objective can be written:

 $$ \begin{array}{r l}&{\mathcal{J}(\varphi,\theta)=\mathbb{E}_{s\sim p^{\pi a l l},(\boldsymbol{i}d s,\boldsymbol{a})\sim\pi^{a l l}}\left[\pi^{a l l}(\boldsymbol{i}d s,\boldsymbol{a}|s)Q^{\pi^{a l l}}(s,\boldsymbol{a})\right]}\\ &{=\mathbb{E}_{p^{\pi a l l},\pi^{a l l}}\left[\prod_{i=1}^{N}\rho_{\varphi}^{i}(\boldsymbol{i}d s^{i}|o^{i})\pi_{\theta}^{i}(a^{i}|o^{i},\boldsymbol{i}d s^{i})Q^{\pi^{i}}(o^{i},a^{i})\right],}\end{array} $$ 

where  $ p^{\pi^{all}} $ denotes the stationary distribution induced by policy  $ \pi^{all} $,  $ Q^{\pi^{all}} $ is the joint action-value function, and  $ Q^{\pi^{i}} $ is the individual action-value function. Here,  $ Q^{\pi^{i}} $ simplifies the learning process by focusing on local state-action pairs rather than the global state, effectively mitigating the scalability issue in large-scale environments.

4.2.2 Optimizing CoS Policy. The CoS policy  $ \rho_{t} $ aims to choose optimal collaborators for intersections. We follow two rules: (a) Intersections should primarily focus on their own decisions. (b) They should also account for and collaborate with each other's decisions. Thus, two key constraints are introduced to ensure these rules in the collaborator selection process.

Firstly, we can denote the distributions of  $ \rho_t $ for all intersection as a collaborator matrix  $ \boldsymbol{M}^\rho \in \mathbb{R}^{N \times N} $, where each element  $ M_{ij}^\rho $ is the matrix signifies the probability of collaborator selection between the intersections  $ i $ and  $ j $.

Rule 1: “You are your biggest collaborator”. Diagonal maximization constraint is imposed. To prioritize its decision-making for each intersection, we enforce the diagonal maximization constraint on the matrix  $ M^{\rho} $, ensuring it remains a valid probability distribution matrix. The main objective is to maximize the sum of the diagonal elements while complying with specific constraints that require the elements to be non-negative, and the row sums must equal to one:

 $$ \max\sum_{i=1}^{N}M_{ii}^{\rho}\ s.t.\ M_{ij}^{\rho}\geq0,\forall i,j;\ \sum_{j=1}^{N}M_{ij}^{\rho}=1,\forall i. $$ 

Rule 2: “Collaboration should be mutually reciprocal”. Symmetry constraint is enforced. To encourage collaboration and mutual consideration between intersections, we incorporate a symmetry constraint into the training. The symmetry constraint, calculated as the mean squared difference between the matrix  $ M^{\rho} $ and its transpose  $ [M^{\rho}]^{T} $, guides the neural network to learn symmetric collaborator selection. Mathematically, the symmetry constraint is formulated as follows:

 $$ \min\frac{1}{N^{2}}\sum_{i=1}^{N}\sum_{j=1}^{N}\left(M_{i j}^{\rho}-\left[M_{i j}^{\rho}\right]^{\mathrm{T}}\right)^{2}. $$ 

In summary, the loss function of optimizing the CoS policy is:

 $$ \mathcal{L}(\rho_{\varphi})=-\mathcal{J}(\varphi,\theta)-\sum_{i=1}^{N}M_{i i}^{\rho}+\frac{1}{N^{2}}\sum_{i=1}^{N}\sum_{j=1}^{N}\left(M_{i j}^{\rho}-\left[M_{i j}^{\rho}\right]^{\mathrm{T}}\right)^{2}. $$ 

Following the multi-agent policy gradient optimization [12, 63], we can derive the gradient for  $ \rho_{\varphi} $:

 $$ \begin{aligned}&\nabla_{\varphi^{i}}\mathcal{J}(\varphi,\theta)=\int_{S}p^{\pi^{all}}(s)\sum_{ids}\rho_{\varphi}(\boldsymbol{i}ds|s)\cdot\\ &\left[\nabla_{\varphi}\sum_{i}\log\rho_{\varphi^{i}}^{i}(\boldsymbol{i}ds^{i}|s)\cdot\sum_{\boldsymbol{a}}\pi(\boldsymbol{a}|s,\boldsymbol{i}ds)\right]\cdot Q_{\pi^{i}}(o^{i},a^{i})ds\\ &-\frac{1}{\mathrm{N}}\nabla\varphi M^{\rho}+\frac{2}{\mathrm{N}}\left(M^{\rho}-[M^{\rho}]^{\mathrm{T}}\right)\nabla\varphi M^{\rho}\\ \end{aligned} $$ 

Algorithm 1 The optimization process.

1: Ensure: Collaborator assignment policy  $ \{\rho^i\}_{i=1}^N $, collaborator-based multi-agent actor  $ \{\pi^i\}_{i=1}^N $, and critics  $ \{Q^i\}_{i=1}^N $;
Initialize:  $ \gamma \mathcal{D} \leftarrow \emptyset, L_1, L_2, B; // L_1, L_2 $ are intervals; B is the mini-batch size.

2: Initialize: the parameters  $ \theta^i, \varphi^i, \phi^i $ for  $ \rho^i, \pi^i $, and  $ Q^i $;

3: for each episode do

4: Reset state  $ \leftarrow \{o_0^i\}_{i=1}^N $; // drop i as  $ o_0 $ for brevity;

5: for each timestep t do

6: Obtain the dual-features  $ e_{ir} $;

7: for each intersection i do

8: Get the collaborator indices  $ ids^i = \rho^i(\cdot | e_{ir}) $ for each intersection;

9: Query the collaborator representation  $ \tilde{e}_{team}^i $ with  $ e_{ir} $ and  $ ids^i $;

10: Get the action  $ a^i = \pi^i(\cdot | [e_{ir}^i || \tilde{e}_{team}^i]) $;

11: end for

12: Take joint actions  $ a_t $;

13: Receive reward  $ r_t $ and observe the next state  $ \mathbf{o}_{t+1} $;

14: Add transition  $ \{o_t, \text{id}s_t, a_t, r_t, o_{t+1}\} $ into  $ \mathcal{D} $;

15: end for

16: if episodes  $ \geq L_1 $ then

17: Sample batch  $ \{o_j, \text{ids}_j, a_j, r_j, \mathbf{o}_{j+1}\}_{j=0}^B \sim \mathcal{D} $;

18: Update the CoS policy  $ \rho^i $ using (11);

19: Update the decision policy  $ \pi^i $ using (12);

20: Update the critic  $ \phi^i $ using (13);

21: end if

22: Updating the target networks every  $ L_2 $ episodes;

23: end for

Thus, incorporating these constraints into the CoS policy gradient optimization enables the policy  $ \rho_{\varphi} $ to learn the optimal collaborator composition progressively. As a result, the policy can promote cooperative decision-making among intersections, resulting in enhanced traffic signal control and improved traffic flow efficiency.

4.2.3 Optimizing Decision Policy. Assumed we have  $ ids \sim \rho_t $ to identify the most appropriate collaborators and then to optimize the multi-agent decision policy  $ \pi_t $ as follows. For the decision policy, we can derive its gradient with the Eq. (7).

 $$ \begin{aligned}&\nabla_{\theta^{i}}\mathcal{J}(\varphi,\theta)=\int_{S}p^{\pi^{all}}(s)\sum_{ids}\rho(ids|s)\sum_{a}\pi_{\theta}(a|s,ids)\cdot\\&\nabla_{\theta}\sum_{i}\log\pi_{\theta^{i}}^{i}(a^{i}|o^{i},ids^{i})\cdot Q_{\pi^{i}}(o^{i},a^{i})ds\\&\approx\mathbb{E}_{(s,a,ids)\sim\mathcal{D}}\left[\nabla_{\theta}\sum_{i}\log\pi_{\theta^{i}}^{i}(a^{i}|o^{i},ids^{i})\cdot Q_{\pi^{i}}(o^{i},a^{i})\right]\\&=\mathbb{E}_{(s,a,ids)\sim D}\left[\sum_{i}\frac{\nabla_{\theta^{i}}\pi_{\theta^{i}}^{i}(a^{i}|o^{i},ids^{i})}{\pi_{\overline{\theta}^{i}}^{i}(a^{i}|o^{i},ids^{i})}\cdot Q_{\pi^{i}}(o^{i},a^{i})\right],\\ \end{aligned} $$ 

where  $ \pi_{\overline{\alpha}^{i}}^{i} $ is the old decision policy used for sampling.

The critic is updated to minimize the difference between the predicted and actual returns, which resembles the action-value TD-learning [38]. The loss function for  $ Q^{\pi^{i}} $ is formulated as follows.

 $$ \mathcal{L}_{Q^{\pi^{i}}}\left(\phi\right)=\mathbb{E}_{\left(o^{i},a^{i},r^{i},o^{i^{\prime}}\right)\sim\mathcal{D}}\left[\left(y^{i}-Q^{\pi^{i}}\left(o^{i},a^{i};\phi\right)\right)^{2}\right], $$ 

where  $ y^i = r^i + \gamma \max_{a^i'} \widetilde{Q}^{\pi^i}(o^i', a^i') $ is the learning target,  $ \gamma $ is the discounted factor, and  $ \widetilde{Q}^{\pi^i} $ is the target network for intersection  $ i $.

The joint optimization scheme ensures the CoS policy and the collaborator-based decision policy converge in the same direction by optimizing the same objective function, which is to maximize the cumulative discounted return in an end-to-end manner. The proposed method facilitates effective decision-making and collaboration among intersections, allowing the two policies to work together harmoniously toward achieving the overall goal.

## 4.3 Algorithmic Framework

Here we provide a detailed pseudocode to elaborate the overall inference and training process. In the inference process, we first select suitable teammate IDs  $ ids^{i} $ for each intersection i based on its observations  $ o^{i} $. Then we calculate the teammate vectors  $ \hat{e}_{team}^{i} $ according to the teammate IDs and concatenate them to the self vector  $ \hat{e}_{ir}^{i} $ to obtain the action for decision making, and finally interact with the SUMO environment. In training, we use the derived loss for backpropagation training. The detailed process is presented in Algorithm 1.

# 5 EXPERIMENTS

### 5.1 Environments

The evaluation scenarios come from the Simulation of Urban Mobility (SUMO) $ ^{1} $, which contains three synthetic scenarios and two real road network maps of different scales, including Grid  $ 4 \times 4 $, Avenue  $ 4 \times 4 $, Grid  $ 5 \times 5 $, Cologne8, and Nanshan. As for each scenario, each individual episode lasts for a time span of 3600 seconds, during which the action interval is  $ \Delta t = 15 $ seconds.

In Table 1, we present detailed statistics about these scenarios. In real-world traffic scenarios, many intersections don't conform to a standard four-arm structure, potentially having varied lane counts and orientations. To ensure the broad application of our method across diverse scenarios, we intentionally conducted experiments in two settings that feature non-standard intersections: Cologne8 and Nanshan.

Dataset Country #Total Int. #2-arm #3-arm #4-arm Flow Type min Flow (/hour) max Flow mean Flow

Grid 4 × 4 synthetic 16 0 0 16 multi-modal Gaussian (m.m.G) 66 136 94.5

Avenue 4 × 4 synthetic 16 0 0 16 m.m.G 94.5 666 364.6

Grid 5 × 5 synthetic 25 0 0 25 m.m.G 120 1363 269.8

Cologned Germany 8 1 3 4 real flow, morning peak (8AM-9AM) 134 212 139.0

Nanshan China 29 1 6 22 real flow, evening peak (6PM-7PM) 160 1440 172.6

<div style="text-align: center;">Table 1: The detailed statistics about evaluation scenarios</div>


Moreover, the network design and hyper-parameter settings are provided in Appendix A.

### 5.2 Baselines

We analyze the performance of our method by comparing it with two conventional transportation techniques and six state-of-the-art (SOTA) RL/MARL algorithms.

Conventional Methods:

• Fixed Time Control (FTC) [33] with random offset executes each phase within a loop, utilizing a pre-defined phase duration.

• MaxPressure [19, 43] greedily chooses the phase with the maximum pressure, which is a SOTA transportation control method.

<div style="text-align: center;">CoSLight: Co-optimizing Collaborator Selection and Decision-making to Enhance Traffic Signal Control</div>



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td rowspan="2">Methods</td><td colspan="5">Average Trip Time (seconds)</td><td colspan="5">Average Delay Time (seconds)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Grid 4×4</td><td style='text-align: center; word-wrap: break-word;'>Avenue 4×4</td><td style='text-align: center; word-wrap: break-word;'>Grid 5×5</td><td style='text-align: center; word-wrap: break-word;'>Cologne8</td><td style='text-align: center; word-wrap: break-word;'>Nanshan</td><td style='text-align: center; word-wrap: break-word;'>Grid 4×4</td><td style='text-align: center; word-wrap: break-word;'>Avenue 4×4</td><td style='text-align: center; word-wrap: break-word;'>Grid 5×5</td><td style='text-align: center; word-wrap: break-word;'>Cologne8</td><td style='text-align: center; word-wrap: break-word;'>Nanshan</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FTC</td><td style='text-align: center; word-wrap: break-word;'>206.68  $ \pm $ 0.54</td><td style='text-align: center; word-wrap: break-word;'>828.38  $ \pm $ 8.17</td><td style='text-align: center; word-wrap: break-word;'>550.38  $ \pm $ 8.31</td><td style='text-align: center; word-wrap: break-word;'>124.4  $ \pm $ 1.99</td><td style='text-align: center; word-wrap: break-word;'>729.02  $ \pm $ 37.03</td><td style='text-align: center; word-wrap: break-word;'>94.64  $ \pm $ 0.43</td><td style='text-align: center; word-wrap: break-word;'>1234.30  $ \pm $ 6.50</td><td style='text-align: center; word-wrap: break-word;'>790.18  $ \pm $ 7.96</td><td style='text-align: center; word-wrap: break-word;'>62.38  $ \pm $ 2.95</td><td style='text-align: center; word-wrap: break-word;'>561.69  $ \pm $ 37.09</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MaxPressure</td><td style='text-align: center; word-wrap: break-word;'>175.97  $ \pm $ 0.70</td><td style='text-align: center; word-wrap: break-word;'>686.12  $ \pm $ 9.57</td><td style='text-align: center; word-wrap: break-word;'>274.15  $ \pm $ 15.23</td><td style='text-align: center; word-wrap: break-word;'>95.96  $ \pm $ 1.11</td><td style='text-align: center; word-wrap: break-word;'>720.89  $ \pm $ 29.94</td><td style='text-align: center; word-wrap: break-word;'>64.01  $ \pm $ 0.71</td><td style='text-align: center; word-wrap: break-word;'>952.53  $ \pm $ 12.48</td><td style='text-align: center; word-wrap: break-word;'>240.00  $ \pm $ 18.43</td><td style='text-align: center; word-wrap: break-word;'>31.93  $ \pm $ 1.07</td><td style='text-align: center; word-wrap: break-word;'>553.94  $ \pm $ 32.61</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>IPPO</td><td style='text-align: center; word-wrap: break-word;'>167.62  $ \pm $ 2.42</td><td style='text-align: center; word-wrap: break-word;'>431.31  $ \pm $ 28.55</td><td style='text-align: center; word-wrap: break-word;'>259.28  $ \pm $ 9.55</td><td style='text-align: center; word-wrap: break-word;'>90.87  $ \pm $ 0.40</td><td style='text-align: center; word-wrap: break-word;'>743.69  $ \pm $ 38.9</td><td style='text-align: center; word-wrap: break-word;'>56.38  $ \pm $ 1.46</td><td style='text-align: center; word-wrap: break-word;'>914.58  $ \pm $ 36.90</td><td style='text-align: center; word-wrap: break-word;'>243.58  $ \pm $ 9.29</td><td style='text-align: center; word-wrap: break-word;'>26.82  $ \pm $ 0.43</td><td style='text-align: center; word-wrap: break-word;'>577.99  $ \pm $ 42.22</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MAPPO</td><td style='text-align: center; word-wrap: break-word;'>164.96  $ \pm $ 1.87</td><td style='text-align: center; word-wrap: break-word;'>565.67  $ \pm $ 44.8</td><td style='text-align: center; word-wrap: break-word;'>300.90  $ \pm $ 8.31</td><td style='text-align: center; word-wrap: break-word;'>97.68  $ \pm $ 2.03</td><td style='text-align: center; word-wrap: break-word;'>744.47  $ \pm $ 30.07</td><td style='text-align: center; word-wrap: break-word;'>53.65  $ \pm $ 1.00</td><td style='text-align: center; word-wrap: break-word;'>1185.2  $ \pm $ 167.48</td><td style='text-align: center; word-wrap: break-word;'>346.78  $ \pm $ 28.25</td><td style='text-align: center; word-wrap: break-word;'>33.37  $ \pm $ 1.97</td><td style='text-align: center; word-wrap: break-word;'>580.49  $ \pm $ 33.6</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MAT</td><td style='text-align: center; word-wrap: break-word;'>246.13  $ \pm $ 24.23</td><td style='text-align: center; word-wrap: break-word;'>421.85  $ \pm $ 73.13</td><td style='text-align: center; word-wrap: break-word;'>356.81  $ \pm $ 11.05</td><td style='text-align: center; word-wrap: break-word;'>111.59  $ \pm $ 18.82</td><td style='text-align: center; word-wrap: break-word;'>754.28  $ \pm $ 58.70</td><td style='text-align: center; word-wrap: break-word;'>106.70  $ \pm $ 14.07</td><td style='text-align: center; word-wrap: break-word;'>565.42  $ \pm $ 91.35</td><td style='text-align: center; word-wrap: break-word;'>217.93  $ \pm $ 40.64</td><td style='text-align: center; word-wrap: break-word;'>25.23  $ \pm $ 8.69</td><td style='text-align: center; word-wrap: break-word;'>415.84  $ \pm $ 75.59</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FRAP</td><td style='text-align: center; word-wrap: break-word;'>161.58  $ \pm $ 1.9</td><td style='text-align: center; word-wrap: break-word;'>383.71  $ \pm $ 4.42</td><td style='text-align: center; word-wrap: break-word;'>238.41  $ \pm $ 10.66</td><td style='text-align: center; word-wrap: break-word;'>88.61  $ \pm $ 0.33</td><td style='text-align: center; word-wrap: break-word;'>709.18  $ \pm $ 21.46</td><td style='text-align: center; word-wrap: break-word;'>50.02  $ \pm $ 0.93</td><td style='text-align: center; word-wrap: break-word;'>794.13  $ \pm $ 42.52</td><td style='text-align: center; word-wrap: break-word;'>203.95  $ \pm $ 8.92</td><td style='text-align: center; word-wrap: break-word;'>27.5  $ \pm $ 0.24</td><td style='text-align: center; word-wrap: break-word;'>542.43  $ \pm $ 21.51</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MPLight</td><td style='text-align: center; word-wrap: break-word;'>179.51  $ \pm $ 0.95</td><td style='text-align: center; word-wrap: break-word;'>541.29  $ \pm $ 45.24</td><td style='text-align: center; word-wrap: break-word;'>261.76  $ \pm $ 6.60</td><td style='text-align: center; word-wrap: break-word;'>98.44  $ \pm $ 0.62</td><td style='text-align: center; word-wrap: break-word;'>668.81  $ \pm $ 7.92</td><td style='text-align: center; word-wrap: break-word;'>67.52  $ \pm $ 0.97</td><td style='text-align: center; word-wrap: break-word;'>1083.18  $ \pm $ 63.38</td><td style='text-align: center; word-wrap: break-word;'>213.78  $ \pm $ 14.44</td><td style='text-align: center; word-wrap: break-word;'>34.38  $ \pm $ 0.63</td><td style='text-align: center; word-wrap: break-word;'>494.05  $ \pm $ 7.52</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>CoLight</td><td style='text-align: center; word-wrap: break-word;'>163.52  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>409.93  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>242.37  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>89.72  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>608.01  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>51.58  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>776.61  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>248.32  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>25.56  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>428.95  $ \pm $ 0.00</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Advanced-CoLight</td><td style='text-align: center; word-wrap: break-word;'>171.63  $ \pm $ 1.71</td><td style='text-align: center; word-wrap: break-word;'>421.44  $ \pm $ 5.61</td><td style='text-align: center; word-wrap: break-word;'>237.67  $ \pm $ 3.02</td><td style='text-align: center; word-wrap: break-word;'>91.22  $ \pm $ 1.01</td><td style='text-align: center; word-wrap: break-word;'>612.34  $ \pm $ 9.79</td><td style='text-align: center; word-wrap: break-word;'>52.31  $ \pm $ 0.01</td><td style='text-align: center; word-wrap: break-word;'>763.78  $ \pm $ 14.01</td><td style='text-align: center; word-wrap: break-word;'>242.50  $ \pm $ 5.06</td><td style='text-align: center; word-wrap: break-word;'>25.12  $ \pm $ 1.08</td><td style='text-align: center; word-wrap: break-word;'>512.45  $ \pm $ 6.98</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MetaGAT</td><td style='text-align: center; word-wrap: break-word;'>165.23  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>374.80  $ \pm $ 0.87</td><td style='text-align: center; word-wrap: break-word;'>266.60  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>90.74  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>676.42  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>53.20  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>772.36  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>234.80  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>26.85  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>503.42  $ \pm $ 0.00</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DuaLight</td><td style='text-align: center; word-wrap: break-word;'>161.0  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>396.65  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>221.83  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>89.74  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>609.89  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>49.32  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>756.99  $ \pm $ 69.44</td><td style='text-align: center; word-wrap: break-word;'>237.71  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>25.35  $ \pm $ 0.00</td><td style='text-align: center; word-wrap: break-word;'>429.49  $ \pm $ 0.00</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>CoSLight</td><td style='text-align: center; word-wrap: break-word;'>159.1  $ \pm $ 3.12</td><td style='text-align: center; word-wrap: break-word;'>364.21  $ \pm $ 4.78</td><td style='text-align: center; word-wrap: break-word;'>220.32  $ \pm $ 1.71</td><td style='text-align: center; word-wrap: break-word;'>90.46  $ \pm $ 0.55</td><td style='text-align: center; word-wrap: break-word;'>621.23  $ \pm $ 7.17</td><td style='text-align: center; word-wrap: break-word;'>46.11  $ \pm $ 1.21</td><td style='text-align: center; word-wrap: break-word;'>744.98  $ \pm $ 16.49</td><td style='text-align: center; word-wrap: break-word;'>178.54  $ \pm $ 4.34</td><td style='text-align: center; word-wrap: break-word;'>24.79  $ \pm $ 1.23</td><td style='text-align: center; word-wrap: break-word;'>476.57  $ \pm $ 17.12</td></tr></table>


<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td rowspan="2">Methods</td><td colspan="5">Average Rewards</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Grid 4×4</td><td style='text-align: center; word-wrap: break-word;'>Avenue 4×4</td><td style='text-align: center; word-wrap: break-word;'>Grid 5×5</td><td style='text-align: center; word-wrap: break-word;'>Cologne8</td><td style='text-align: center; word-wrap: break-word;'>Nanshan</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FTC</td><td style='text-align: center; word-wrap: break-word;'>-0.614  $ \pm $ 0.015</td><td style='text-align: center; word-wrap: break-word;'>-4.503  $ \pm $ 0.025</td><td style='text-align: center; word-wrap: break-word;'>-2.346  $ \pm $ 0.052</td><td style='text-align: center; word-wrap: break-word;'>-2.114  $ \pm $ 0.021</td><td style='text-align: center; word-wrap: break-word;'>-3.479  $ \pm $ 0.186</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MaxPressure</td><td style='text-align: center; word-wrap: break-word;'>-0.393  $ \pm $ 0.003</td><td style='text-align: center; word-wrap: break-word;'>-4.032  $ \pm $ 0.040</td><td style='text-align: center; word-wrap: break-word;'>-1.132  $ \pm $ 0.013</td><td style='text-align: center; word-wrap: break-word;'>-0.756  $ \pm $ 0.012</td><td style='text-align: center; word-wrap: break-word;'>-3.055  $ \pm $ 0.162</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>IPPO</td><td style='text-align: center; word-wrap: break-word;'>-0.336  $ \pm $ 0.004</td><td style='text-align: center; word-wrap: break-word;'>-2.558  $ \pm $ 0.213</td><td style='text-align: center; word-wrap: break-word;'>-0.943  $ \pm $ 0.037</td><td style='text-align: center; word-wrap: break-word;'>-0.646  $ \pm $ 0.015</td><td style='text-align: center; word-wrap: break-word;'>-3.555  $ \pm $ 0.097</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MAPPO</td><td style='text-align: center; word-wrap: break-word;'>-0.308  $ \pm $ 0.006</td><td style='text-align: center; word-wrap: break-word;'>-2.744  $ \pm $ 0.238</td><td style='text-align: center; word-wrap: break-word;'>-1.273  $ \pm $ 0.107</td><td style='text-align: center; word-wrap: break-word;'>-1.697  $ \pm $ 0.132</td><td style='text-align: center; word-wrap: break-word;'>-4.161  $ \pm $ 0.195</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MAT</td><td style='text-align: center; word-wrap: break-word;'>-0.328  $ \pm $ 0.001</td><td style='text-align: center; word-wrap: break-word;'>-3.100  $ \pm $ 0.279</td><td style='text-align: center; word-wrap: break-word;'>-1.109  $ \pm $ 0.132</td><td style='text-align: center; word-wrap: break-word;'>-1.811  $ \pm $ 0.117</td><td style='text-align: center; word-wrap: break-word;'>-5.002  $ \pm $ 0.755</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FRAP</td><td style='text-align: center; word-wrap: break-word;'>-0.274  $ \pm $ 0.000</td><td style='text-align: center; word-wrap: break-word;'>-2.573  $ \pm $ 0.012</td><td style='text-align: center; word-wrap: break-word;'>-1.064  $ \pm $ 0.002</td><td style='text-align: center; word-wrap: break-word;'>-0.705  $ \pm $ 0.020</td><td style='text-align: center; word-wrap: break-word;'>-3.191  $ \pm $ 0.170</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MPLight</td><td style='text-align: center; word-wrap: break-word;'>-0.414  $ \pm $ 0.012</td><td style='text-align: center; word-wrap: break-word;'>-4.079  $ \pm $ 0.049</td><td style='text-align: center; word-wrap: break-word;'>-1.087  $ \pm $ 0.041</td><td style='text-align: center; word-wrap: break-word;'>-0.842  $ \pm $ 0.026</td><td style='text-align: center; word-wrap: break-word;'>-3.117  $ \pm $ 0.116</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>CoLight</td><td style='text-align: center; word-wrap: break-word;'>-0.309  $ \pm $ 0.006</td><td style='text-align: center; word-wrap: break-word;'>-2.326  $ \pm $ 0.057</td><td style='text-align: center; word-wrap: break-word;'>-0.918  $ \pm $ 0.042</td><td style='text-align: center; word-wrap: break-word;'>-0.695  $ \pm $ 0.008</td><td style='text-align: center; word-wrap: break-word;'>-2.939  $ \pm $ 0.092</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Advanced-CoLight</td><td style='text-align: center; word-wrap: break-word;'>-0.291  $ \pm $ 0.011</td><td style='text-align: center; word-wrap: break-word;'>-2.317  $ \pm $ 0.018</td><td style='text-align: center; word-wrap: break-word;'>-1.112  $ \pm $ 0.018</td><td style='text-align: center; word-wrap: break-word;'>-0.607  $ \pm $ 0.001</td><td style='text-align: center; word-wrap: break-word;'>-2.904  $ \pm $ 0.016</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MetaGAT</td><td style='text-align: center; word-wrap: break-word;'>-0.468  $ \pm $ 0.126</td><td style='text-align: center; word-wrap: break-word;'>-2.538  $ \pm $ 0.077</td><td style='text-align: center; word-wrap: break-word;'>-1.326  $ \pm $ 0.311</td><td style='text-align: center; word-wrap: break-word;'>-0.805  $ \pm $ 0.168</td><td style='text-align: center; word-wrap: break-word;'>-3.289  $ \pm $ 0.261</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DuaLight</td><td style='text-align: center; word-wrap: break-word;'>-0.331  $ \pm $ 0.112</td><td style='text-align: center; word-wrap: break-word;'>-2.711  $ \pm $ 0.005</td><td style='text-align: center; word-wrap: break-word;'>-1.007  $ \pm $ 0.622</td><td style='text-align: center; word-wrap: break-word;'>-0.724  $ \pm $ 0.338</td><td style='text-align: center; word-wrap: break-word;'>-4.330  $ \pm $ 0.306</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>CoSLight</td><td style='text-align: center; word-wrap: break-word;'>-0.251  $ \pm $ 0.000</td><td style='text-align: center; word-wrap: break-word;'>-2.309  $ \pm $ 0.068</td><td style='text-align: center; word-wrap: break-word;'>-0.890  $ \pm $ 0.012</td><td style='text-align: center; word-wrap: break-word;'>-0.538  $ \pm $ 0.007</td><td style='text-align: center; word-wrap: break-word;'>-2.899  $ \pm $ 0.014</td></tr></table>

Table 3: Intersection-wise evaluation. Our method constantly achieves the best performance.

##### RL-based Methods:

• IPPO [1, 2] controls each intersection with an independent PPO agent, which is trained with the data from the current intersection.

• MAPPO [37, 59] executes with an independent PPO agent and is trained collectively using the data from all intersections, enabling an optimized coordinated traffic flow.

• MAT [55] is a strong baseline in MARL with centralized training with centralized execution paradigm, modeling the TSC as a sequential problem.

• FRAP [66] models phase competition and employs deep Q-network agent for each intersection to optimize traffic phase operation.

• MPLight [6] utilizes the concept of pressure as both state and reward to coordinate multiple intersections, which is based on FRAP.

• CoLight [52] leverages a GAT to extract neighboring information, thereby assisting the agent in optimizing queue length.

• Advanced-CoLight [65] combines advanced traffic state for the traffic movement representation with a pressure of queuing and demand of running for vehicles with CoLight, to enhance the decision-making process.

• MetaGAT [24] leverages GAT-based context to boost cooperation among intersections.

• DuaLight [25] introduces a scenario-specific experiential weight module and a scenario-shared co-train module to facilitate the information extraction of scenarios and intersections.

## 5.3 Evaluation Metrics

We utilize two evaluation metrics in our study. Firstly, at the scenario level [2], we compute the average delay, and average trip time by tracking vehicles in the scenario. Specifically, delay signifies the holdup caused by signalized intersections (either stop or approach delay) for a vehicle, and trip time denotes the complete duration of a vehicle's journey from its origin to its destination.

Secondly, at the intersection level, we employ the external reward of the environment as an evaluation criterion, including the average delay time, average wait time, average queue length, and average pressure for each intersection. These metrics are calculated at each individual intersection by averaging the values across all vehicles.

## 5.4 Main Results

Scenario-wise evaluation. As illustrated in Table 2, the performances marked in boldface and underlined represent the best and second-best results, respectively. CoSLight consistently achieves substantial performance improvements, reducing the average delay time by 7.68%, the average trip time by 1.98%, which not only validates the effectiveness of CoSLight but also highlights its potential to efficiently manage and enhance multi-intersection collaboration in various traffic scenarios.

Intersection-wise evaluation. Table 3 shows CoSLight achieves the best results in all scenarios. Compared to the second-best result, ours achieved a 7.71 % improvement on average. This consistent performance enhancement across intersection-wise evaluation metrics underlines the robustness of our proposed method.

Notably, these findings provide strong evidence that our algorithm performs well not only in terms of global cooperation (scenario-wise evaluation) but also from the perspective of benefits at individual intersections (intersection-wise evaluation). This dual-level efficacy showcases the effectiveness of our approach, signifying its ability to foster overall road network performance while simultaneously optimizing individual intersection operations.

## 5.5 Ablation Analysis of Three Settings

In this section, we will examine the effects of our designed components, the collaborator matrix, and the constraints on the collaborator matrix.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_3/imgs/img_in_chart_box_114_278_274_388.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A27Z%2F-1%2F%2F99e013b547bea6ab7593e2f489243189c57a71f92ba64ad51ff9d2bf685c163b" alt="Image" width="13%" /></div>


<div style="text-align: center;">(a) Ablating FRAP, Transformer, Teammate CoS; Replacing CoS, Dual Extractor with GNNs in Grid 5×5</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_3/imgs/img_in_chart_box_278_279_430_388.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A27Z%2F-1%2F%2F0753d7f1adcf68172573f8e38ee7101a75291813421acb2621f3a313e4a59a91" alt="Image" width="12%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_3/imgs/img_in_chart_box_431_277_583_388.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A27Z%2F-1%2F%2F04146b62d4ca36c086405548d6a8cf72976e7e685f724e222a096eabad27e6d6" alt="Image" width="12%" /></div>


<div style="text-align: center;">(c) Ablating diagonal, symmetry, and both constraints in collaborator matrix in Cologne8</div>


<div style="text-align: center;">Figure 5: Ablation Studies</div>


Ablation of Components. Firstly, we selectively remove different modules, specifically the FRAP, Transformer, and Collaborator modules, to validate the necessity of each component in CoSLight. In Figure 5(a), (1) it is evident that the Transformer module that aggregates other intersections' information and the Teammate CoS module that adaptively selects collaborators are the most important. (2) Moreover, to justify the MLP design for CoS, we replace it in CoS with GNNs, where we could observe a significant slowdown of convergence and performance drop in CoS with GNN, due to GNN's computational complexity as the bottleneck. (3) To justify Transformer as a better design to aggregate other intersections' information, we replaced the transformer module in the dual extractor with GNNs (Dual with GNN). We see similar performance drops. Such a feature extractor encourages learning better embedding for each agent and understanding them better, shown in Figure 6.

Ablation of the Collaborator Matrix. To further assess the impact of the co-learned collaborator matrix, we conduct experiments where the matrix in the proposed CoS is replaced with both fixed (as topological adjacency matrix) and random (freezing the collaborator selection after randomly initializing top-k selection) matrices. Figure 5(b) shows that using co-learned collaborator matrices in CoS can boost performance, highlighting the critical role of the dynamical collaboration matrix in achieving effective coordination. Thus, the joint optimization of the collaborator matrix with decision policies is key for optimizing cumulative rewards.

Ablation of Constraints on Collaborator Matrix. In Figure 5(c), we further evaluate the contributions of the constraints in Eq (8) and (9) by removing the Diagonal constraint, the Symmetry constraint, or both. Removing the Symmetry constraint significantly degrades performance, which underlines the symmetric interplay between each other is essential. Conversely, the Diagonal constraint has a marginal impact, primarily enhancing the convergence speed. These insights highlight the value of the Symmetry constraint for optimality and the Diagonal constraint for efficiency.

In summary, the ablations provide empirical evidence that each dimension of CoS is vital. The co-learning of the collaborator matrix, the adherence to specific constraints, and the integration of crucial components such as the FRAP, Transformer, and Collaborator modules all contribute to the robustness and effectiveness of the system. Through validation, we demonstrate that our model leads to enhanced performance in complex environments.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_3/imgs/img_in_chart_box_649_149_869_316.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A27Z%2F-1%2F%2Ffcc1f9f66b36d9ab181871c4c80039bf1b8c34d0a1c289c4d2ac9cc10964e9e0" alt="Image" width="17%" /></div>


<div style="text-align: center;">(a) Avenue 4 × 4</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_3/imgs/img_in_chart_box_885_150_1104_314.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A27Z%2F-1%2F%2F705a918b8aa08540696ebf6cd578d229c4db9f354fdcd2ac51421bea0e882f11" alt="Image" width="17%" /></div>


<div style="text-align: center;">(b) Nanshan</div>


<div style="text-align: center;">Figure 6: Visualization of Dual-feature. Each color represents a specific intersection. CoSLight has captured unique features at each intersection with distinct clustering patterns.</div>


### 5.6 Visualization Analysis of Dual-feature

We analyze the dual-feature embeddings from our Dual-Feature Extractor. As shown in Figure 6, we test 10 episodes for each intersection in each scenario, resulting in 2400 dual-feature embeddings visualized using the t-SNE technique [41].

These embeddings demonstrate a distinct clustering pattern, suggesting that our model captures unique features at each intersection effectively. This allows the model to group similar states together, adapting to variations in traffic conditions and intersection-level characteristics. This adaptability is a crucial advantage of our approach, contributing significantly to its performance improvement.

In conclusion, through rigorous experiments and insightful analysis, our study confirms that our method, which integrates dual-feature extraction and multi-intersection collaboration, provides an effective and efficient solution for the TSC.

## 5.7 Analysis of Collaborator Number k

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_3/imgs/img_in_chart_box_645_833_871_1000.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A27Z%2F-1%2F%2F57a62573b21d38066cf211e69243603e5acba12bda841c605174bf401e598fbe" alt="Image" width="18%" /></div>


<div style="text-align: center;">(a) Avenue 4 × 4</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_3/imgs/img_in_chart_box_883_834_1104_1013.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A27Z%2F-1%2F%2Fa3c50f3bd4c54e06f49fbb5180fd94d9f06a6efb73a874b9d282141882857b49" alt="Image" width="18%" /></div>


<div style="text-align: center;">(b) Grid 5 × 5</div>


<div style="text-align: center;">Figure 7: Violin plots display the performance trade-offs at varying numbers of collaborators.</div>


Figure 7 shows the trade-off between the number of collaborators and performance. For example, in Avenue 4×4 (Figure 7(a)), k = 8 yields the best results, suggesting an optimal balance between useful information and performance gains. Information from more collaborators beyond this point does not guarantee improved results and may lead to higher resource usage. This finding poses a direction for future work to dynamically determine the ideal number of collaborators, potentially enhancing the algorithm's efficiency.

## 5.8 Visualization of Collaborator Matrix

In this section, we visually analyze the collaborator matrix to offer an intuitive understanding of attention distribution among intersections during training. The results are shown at the start and end of training in the Cologne8 (8 intersections) and Avenue 4 × 4 (16 intersections) scenarios, respectively. The saliency maps in Figure 8

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_4/imgs/img_in_image_box_106_152_342_300.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A28Z%2F-1%2F%2F1eba9268f6d67fd21a76d5660ed5639723729a9faf6a3b94618b3a181c733549" alt="Image" width="19%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_4/imgs/img_in_chart_box_354_151_589_299.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A28Z%2F-1%2F%2Fa24e38c1d7396ad3be199a74555b1a58b69a39fdbc03a825bd0e8bd9d521ab4d" alt="Image" width="19%" /></div>


<div style="text-align: center;">Figure 8: Saliency maps of collaborator matrix. The deeper the color, the stronger the correlation. CoSLight has learned diagonal maximization and symmetry constraints.</div>


depict the state of the collaborator matrix at the beginning and the end of the training, respectively.

Upon the conclusion of the training, we notice a deepening of the color along the diagonal elements of the saliency map. This signifies an increased self-attention, indicating that the intersections have adapted to pay more heed to their own states. Additionally, the symmetry apparent in the saliency map suggests mutual awareness among intersections. As the training progresses, intersections not only learn to focus on themselves but also pay attention to their peers, signifying a learned mutual collaboration.

These observations validate the effectiveness of our approach in creating a collaborative environment among intersections, thus leading to enhanced performance.

## 5.9 Visualization of Collaborator Selection

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_4/imgs/img_in_image_box_119_755_330_908.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A28Z%2F-1%2F%2F32609527918355ccf2957211c5fb43bd9fbce0afeedf4f46a0029255f375bdc0" alt="Image" width="17%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_4/imgs/img_in_image_box_337_752_579_852.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A28Z%2F-1%2F%2F153943e0f75962ca18f2d69e21dfd5796fb0d0790e5cb1ed9e408413b49562d6" alt="Image" width="19%" /></div>


<div style="text-align: center;">(a) The topological layout of Cologne 8, emphasizes 8 signaled intersections and 3 detailed intersections.</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_4/imgs/img_in_chart_box_104_931_345_1122.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A28Z%2F-1%2F%2F63a04ba3b5a277a723af087b2d80d2626639549e2a4aa0ba7fba14b81521ab39" alt="Image" width="19%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//721e6182-6944-4521-9172-5854b2ed1553/markdown_4/imgs/img_in_chart_box_348_932_587_1120.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-02-13T14%3A50%3A28Z%2F-1%2F%2F8e9229159002506e5f7b1a089d8dd78c23a88933258bf83b0d36fb156ea6359a" alt="Image" width="19%" /></div>


<div style="text-align: center;">Figure 9: Collaborator selection on Cologne8, showing self-selection and efficient cross-collaboration at (b) K = 1 with predominantly solo dynamics, and (c) K = 3 with inter-agent collaboration with not just the topological selection.</div>


Figure 9 depicts the collaborator selection process. For K = 1 in Figure 9 (b), self-selection is prevalent; however, agent 0 displays varied collaboration patterns, likely due to the intricacy of its signal control tasks (refer to Figure 9 (a)), which require engaging with multiple collaborators for optimal traffic management. When K increases to 3, as shown in Figure 9 (c), agents exhibit both self-selection and mutual collaboration, forming complex interaction networks. For example, agent 3 largely collaborates with agents 2 (non-neighbor) and 4 (neighbor); Similarly, agent 4 with agents 3 (neighbor) and 6 (neighbor); Also, agent 2 with agents 6 (non-neighbor) and 7 (non-neighbor). Specifically, agent 2 is quite far from 6 and 7, but they form a strong collaboration since agent 2 is in the office building and agents 6 and 7 are in the community residential region.

Overall, such learned patterns suggest strategic selection beyond topological neighbors. Collectively, these results affirm that the collaborator selection mechanism is adaptively responsive to both the complexity of traffic tasks and the benefits of strategic collaboration to optimize traffic flow.

### 5.10 Average Inference and Training Time

We collected 100 episodes over 100 training epochs to obtain the average inference time for CoS and Decision policies per episode and average training time per epoch. We experimented on 4 NVIDIA TITAN Xp GPUs(12G). In Table 4, across five scenarios, CoS inference and training times average 2.83% and 31.29%, respectively, while Decision policy averages 2.66% for inference and 63.21% for training. The CoS strategy adds a 34.42% time overhead, justifiable by its performance benefits.


<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>CoS</td><td style='text-align: center; word-wrap: break-word;'>Decision policy</td><td style='text-align: center; word-wrap: break-word;'>CoS Training</td><td style='text-align: center; word-wrap: break-word;'>Decision Training</td><td style='text-align: center; word-wrap: break-word;'>Total Time</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Grid 4x4</td><td style='text-align: center; word-wrap: break-word;'>0.153 $ \pm $0.008</td><td style='text-align: center; word-wrap: break-word;'>0.143 $ \pm $0.008</td><td style='text-align: center; word-wrap: break-word;'>1.522 $ \pm $0.112</td><td style='text-align: center; word-wrap: break-word;'>2.940 $ \pm $0.163</td><td style='text-align: center; word-wrap: break-word;'>4.758 $ \pm $0.179</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Avenue 4x4</td><td style='text-align: center; word-wrap: break-word;'>0.161 $ \pm $0.006</td><td style='text-align: center; word-wrap: break-word;'>0.152 $ \pm $0.006</td><td style='text-align: center; word-wrap: break-word;'>1.423 $ \pm $0.176</td><td style='text-align: center; word-wrap: break-word;'>3.050 $ \pm $0.203</td><td style='text-align: center; word-wrap: break-word;'>4.785 $ \pm $0.276</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Grid 5x5</td><td style='text-align: center; word-wrap: break-word;'>0.168 $ \pm $0.011</td><td style='text-align: center; word-wrap: break-word;'>0.153 $ \pm $0.011</td><td style='text-align: center; word-wrap: break-word;'>2.531 $ \pm $0.093</td><td style='text-align: center; word-wrap: break-word;'>6.148 $ \pm $0.123</td><td style='text-align: center; word-wrap: break-word;'>9.000 $ \pm $0.123</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Cologne8</td><td style='text-align: center; word-wrap: break-word;'>0.111 $ \pm $0.007</td><td style='text-align: center; word-wrap: break-word;'>0.105 $ \pm $0.007</td><td style='text-align: center; word-wrap: break-word;'>2.001 $ \pm $0.096</td><td style='text-align: center; word-wrap: break-word;'>2.336 $ \pm $0.192</td><td style='text-align: center; word-wrap: break-word;'>4.553 $ \pm $0.186</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Nanshan</td><td style='text-align: center; word-wrap: break-word;'>0.190 $ \pm $0.008</td><td style='text-align: center; word-wrap: break-word;'>0.181 $ \pm $0.008</td><td style='text-align: center; word-wrap: break-word;'>1.321 $ \pm $0.124</td><td style='text-align: center; word-wrap: break-word;'>4.130 $ \pm $0.511</td><td style='text-align: center; word-wrap: break-word;'>5.822 $ \pm $0.507</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Avg (Percent)</td><td style='text-align: center; word-wrap: break-word;'>0.156(2.83%)</td><td style='text-align: center; word-wrap: break-word;'>0.147(2.66%)</td><td style='text-align: center; word-wrap: break-word;'>1.760(31.29%)</td><td style='text-align: center; word-wrap: break-word;'>3.721(63.21%)</td><td style='text-align: center; word-wrap: break-word;'>5.778</td></tr></table>

<div style="text-align: center;">Table 4: Average inference and training time (s).</div>


# 6 CONCLUSION

In this paper, we introduce an innovative approach to traffic signal control, employing a top-k collaborator selection policy with a dual-feature extractor. This unique strategy allows for the effective extraction of phase- and intersection-level representations while adaptively selecting collaborators for enhanced multi-intersection collaboration. Moreover, we are the first to propose a joint optimization regime to train the CoS and decision policies simultaneously for maximizing the cumulative discounted return. Comprehensive experiments on both synthetic and real-world datasets validate our approach's superiority. The extensive analysis further reinforces the effectiveness and efficacy of CoSLight.

Future research could potentially explore an adaptive mechanism to efficiently determine the optimal number of collaborators, thereby enhancing the performance and effectiveness of traffic signal control. Moreover, enhancing the explainability of collaborator selection processes could provide valuable insights, potentially enabling more intuitive and transparent decision-making to promote cooperation.



## Appendices

## A DETAILED NETWORK ARCHITECTURE AND HYPER-PARAMETERS DESCRIPTIONS

There is a summary of all the neural networks used in our framework about the network structure, layers, and activation functions.


<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>Network Structure</td><td style='text-align: center; word-wrap: break-word;'>Layers</td><td style='text-align: center; word-wrap: break-word;'>Hidden Size</td><td style='text-align: center; word-wrap: break-word;'>Activation Functions</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FRAP</td><td style='text-align: center; word-wrap: break-word;'>FRAP $ ^{1} $</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>-</td></tr><tr><td rowspan="3">Transformer Backbone</td><td style='text-align: center; word-wrap: break-word;'>Positional Embedding</td><td style='text-align: center; word-wrap: break-word;'>1</td><td style='text-align: center; word-wrap: break-word;'>64</td><td style='text-align: center; word-wrap: break-word;'>None</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Encoder Layer</td><td style='text-align: center; word-wrap: break-word;'>2</td><td style='text-align: center; word-wrap: break-word;'>64</td><td style='text-align: center; word-wrap: break-word;'>ReLu</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Output Layer (MLP) 3</td><td style='text-align: center; word-wrap: break-word;'>1</td><td style='text-align: center; word-wrap: break-word;'>32</td><td style='text-align: center; word-wrap: break-word;'>ReLu</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Top-k Collaborator Assignment</td><td style='text-align: center; word-wrap: break-word;'>MLP</td><td style='text-align: center; word-wrap: break-word;'>2</td><td style='text-align: center; word-wrap: break-word;'>64</td><td style='text-align: center; word-wrap: break-word;'>ReLu</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Actor</td><td style='text-align: center; word-wrap: break-word;'>MLP+RNN+MLP</td><td style='text-align: center; word-wrap: break-word;'>3</td><td style='text-align: center; word-wrap: break-word;'>$ [64]+[64]+[32] $</td><td style='text-align: center; word-wrap: break-word;'>ReLu</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Critic</td><td style='text-align: center; word-wrap: break-word;'>MLP</td><td style='text-align: center; word-wrap: break-word;'>2</td><td style='text-align: center; word-wrap: break-word;'>64</td><td style='text-align: center; word-wrap: break-word;'>ReLu</td></tr></table>

<div style="text-align: center;">Table A1: The Summary for Network Architecture</div>


There are our hyper-parameter settings for the training, shown in Table A2.


<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Description</td><td style='text-align: center; word-wrap: break-word;'>Value</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>optimizer</td><td style='text-align: center; word-wrap: break-word;'>AdamW</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>learning rate</td><td style='text-align: center; word-wrap: break-word;'>$ 5 \times 10^{-4} $</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>group embedding size</td><td style='text-align: center; word-wrap: break-word;'>32</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>attention head</td><td style='text-align: center; word-wrap: break-word;'>8</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>transformer layer</td><td style='text-align: center; word-wrap: break-word;'>2</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>actor embedding size</td><td style='text-align: center; word-wrap: break-word;'>32</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>state key</td><td style='text-align: center; word-wrap: break-word;'>$ [&#x27;current_phase&#x27;, &#x27;car_num&#x27;, &#x27;queue_length&#x27;, &#x27;occupancy&#x27;, &#x27;flow&#x27;, &#x27;stop_car_num&#x27;, &#x27;pressure&#x27;] $</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>number of actions</td><td style='text-align: center; word-wrap: break-word;'>8</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>interaction steps</td><td style='text-align: center; word-wrap: break-word;'>300000</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>$ \alpha $</td><td style='text-align: center; word-wrap: break-word;'>$ 10^{-4} $</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>$ \beta_{1} $</td><td style='text-align: center; word-wrap: break-word;'>0.9</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>$ \beta_{2} $</td><td style='text-align: center; word-wrap: break-word;'>0.999</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>$ \epsilon $-greedy  $ \epsilon $</td><td style='text-align: center; word-wrap: break-word;'>$ 10^{-5} $</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>clipping  $ \epsilon $</td><td style='text-align: center; word-wrap: break-word;'>0.2</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>seed</td><td style='text-align: center; word-wrap: break-word;'>[0, 10)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>number of process</td><td style='text-align: center; word-wrap: break-word;'>64</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>eval interval</td><td style='text-align: center; word-wrap: break-word;'>4000</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>eval episodes</td><td style='text-align: center; word-wrap: break-word;'>100</td></tr></table>

<div style="text-align: center;">Table A2: The hyper-parameter settings.</div>


By the way, we refer readers to the source code in the supplementary to check the detailed hyper-parameters.

## B THE DETAILS ABOUT FRAP

At the phase level, we adopt FRAP [66] to obtain the phase-wise representation. The raw observations o from the simulator include K features, such as the number of vehicles, queue length, the current phase, the flow, etc. For any traffic movement  $ m, m \in \{1, ..., 8\} $ in an intersection i, the k-th feature in the raw observation can be denoted as  $ o_m, k^i $. For brevity, the superscript i is omitted hereinafter. First, the embedding of traffic movement m is obtained by MultiLayer Perceptron (MLP):

 $$ \boldsymbol{e}_{m}=||_{k=1}^{K}Sigmoid(MLP_{k}(o_{m,k})), $$ 

where || denotes concatenation, and Sigmoid is the activation function. Then FRAP module is applied to extract the phase competition representation, denoted as follows.

 $$ \boldsymbol{e}_{p c r}=F R A P(\boldsymbol{e}_{m_{1}},...,\boldsymbol{e}_{m_{8}}). $$ 

The process can be summarized as follows.

(1) Phase embedding: Each phase $p$ consists of two movements $m_{1}, m_{2}$, and we get the phase embedding $e_{p} = e_{m_{1}} + e_{m_{2}}$.

(2) Phase pair representation: For any pair  $ p_k, p_l $ from different phases, the pairwise relation vector is  $ \mathbf{e}_{p_k, p_l} = \mathbf{e}_{p_k} || \mathbf{e}_{p_l} $. Gathering the vectors of all phase pairs can obtain the pair demand embedding volume  $ E $. Then the phase pair representation can be denoted as  $ \mathbf{e}_{ppr} = \text{Conv}_{1 \times 1}(E) $, where  $ \text{Conv}_{1 \times 1} $ is the convolutional layer with  $ 1 \times 1 $ filters.

(3) Phase competition: Let M be phase competition mask, and the phase competition representation can be obtained by:

 $ e_{pcr} = Conv_{1 \times 1}(e_{ppr} \otimes M) $, where  $ \otimes $ is the element-wise multiplication. Here, we reshape  $ e_{pcr} $ as a vector through flatten operation.

Finally, an MLP is utilized to mine the phase representation in the intersection i as follows.

 $$ \boldsymbol{e}_{pr}^{i}=MLP(\boldsymbol{e}_{pcr}) $$ 

## C ADDITIONAL RESULTS OF HOPS

As shown in Figure 3, we further conducted experiments on Grid 4 × 4 and Grid 5 × 5 to substantiate our assertion: the optimal number and range of collaborating intersections vary across different scenarios. For instance, in Grid 4 × 4, the impact of collaboration remains largely consistent regardless of increasing distances, suggesting that the benefits of collaboration might be distance-independent in this setting. Alternatively, it could indicate that merely selecting topologically adjacent intersections might not enhance the collaborative outcomes. In contrast, Grid 5 × 5 displays a negative impact with one-hop collaboration, whereas two-hop collaboration produces the greatest benefits. This underscores the significance of precisely and judiciously selecting collaborators, highlighting that not just the presence, but the quality and context of collaboration matters.

