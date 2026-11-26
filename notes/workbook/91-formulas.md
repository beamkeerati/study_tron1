<!-- สร้างโดย scripts/workbook_backmatter.py จากเนื้อหาของบทที่ 1-30 ห้ามแก้ไฟล์นี้ด้วยมือ ให้แก้ที่บทแล้วรันสคริปต์ใหม่ -->

## สูตรทั้งหมดในหน้าเดียว

หน้านี้รวมสูตรที่ใช้จริงในหนึ่ง iteration ของ PPO บน Tron1 เรียงตามลำดับที่โค้ดคำนวณ สูตรทุกตัวคัดมาจากบทที่สอน เลขในวงเล็บคือเลขประจำตัวของสมการตามบทนำ ถ้าเขียนสูตรเหล่านี้ได้ครบโดยไม่เปิดเล่ม และบอกได้ว่าแต่ละตัวทำงานเมื่อไร ก็เดิน iteration ของตารางตัวอย่างได้ด้วยมือ

### หนึ่ง iteration เรียงตามเวลา

**ทุก step ตอนเก็บข้อมูล**

| สมการ | สูตร | ทำอะไร | บท |
| --- | --- | --- | --- |
| (9) | $\pi_\theta(\cdot \mid s) = \mathcal{N}\big(\mu_\theta(s),\ \sigma\big), \qquad \sigma = e^{\ell}$ | ประกอบระฆังแล้วสุ่ม action | 11 |
| (16) | $a = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 1)$ |  | 12 |
| (14) | $\log p(a) = -\tfrac{1}{2}\ln 2\pi - \ln\sigma - \frac{(a-\mu)^2}{2\sigma^2}$ | จด $\log\pi_{old}$ ของ action ที่สุ่มได้ ลงตาราง | 10 |
| (15) | $\log\pi_\theta(a \mid s) = \sum_{i=1}^{8} \log\pi(a_i \mid s)$ |  | 11 |
| (17) | $q_{\text{target}} = 0.25\,a + q_{\text{default}}$ | environment แปลง action เป็นมุมเป้าหมายและแรงบิด | 15 |
| (18) | $\tau = k_p\,(q_{\text{target}} - q) - k_d\,\dot q$ |  | 15 |

**ครั้งเดียวต่อตาราง**

| สมการ | สูตร | ทำอะไร | บท |
| --- | --- | --- | --- |
| (19) | $\delta_t = r_t + (1 - \text{done}_t)\,\gamma\,V(s_{t+1}) - V(s_t)$ | critic ทำนายพลาดเท่าไรในหนึ่ง step | 17 |
| (26) | $A_t = \delta_t + (1-\text{done}_t)\,\gamma\lambda\,A_{t+1}, \qquad A_T = 0$ | เดินย้อนจากแถวล่างขึ้นบน สะสมเป็น $A_t$ | 22 |
| (27) | $R_t = A_t + V_t$ | เป้าที่ใช้สอน critic | 22 |
| (28) | $\hat A_t = \frac{A_t - \text{mean}(A)}{\text{std}(A) + 10^{-8}}$ | ปรับ $A_t$ ให้เทียบกันได้ทั้งตาราง | 22 |

**ทุก gradient step**

| สมการ | สูตร | ทำอะไร | บท |
| --- | --- | --- | --- |
| (30) | $D_{KL}\big(\pi_{old} \,\Vert\, \pi_{new}\big) = \sum_{i}\left[\ln\frac{\sigma_{n,i}}{\sigma_{o,i}} + \frac{\sigma_{o,i}^2 + (\mu_{o,i}-\mu_{n,i})^2}{2\,\sigma_{n,i}^2} - \frac12\right]$ | วัดว่าระฆังขยับไปเท่าไร แล้วปรับ $\alpha$ | 26 |
| (37) | ถ้า KL เฉลี่ยเกิน 0.02 หาร $\alpha$ ด้วย 1.5 ถ้าต่ำกว่า 0.005 คูณด้วย 1.5 โดยอยู่ในช่วง $10^{-5}$ ถึง $10^{-2}$ |  | 26 |
| (21) | $\rho_t = \frac{\pi_{new}(a_t \mid s_t)}{\pi_{old}(a_t \mid s_t)} = \exp\!\big(\log\pi_{new}(a_t \mid s_t) - \log\pi_{old}(a_t \mid s_t)\big)$ | ระฆังใหม่ชอบ action เดิมมากขึ้นกี่เท่า | 18 |
| (29) | $L^{CLIP} = \text{mean}\Big[\max\big(-\rho_t\,\hat A_t,\ \ -\text{clip}(\rho_t,\ 1-\epsilon,\ 1+\epsilon)\,\hat A_t\big)\Big]$ | loss ของ policy | 23 |
| (32) | $L_V = \text{mean}\Big[\max\big((V_{new} - R_t)^2,\ (V_{clip} - R_t)^2\big)\Big]$ | loss ของ critic | 24 |
| (31) | $H = \sum_{i}\Big[\tfrac12 + \tfrac12\ln 2\pi + \ln\sigma_i\Big]$ | ความกว้างของระฆัง | 24 |
| (33) | $L = L^{CLIP} + c_V\,L_V - c_H\,H$ | รวมสามพจน์เป็นตัวเลขตัวเดียว | 24 |
| (G) | $\frac{\partial L_i}{\partial\mu} = -\hat A_t\ \rho_t\ \frac{a_t - \mu}{\sigma^2}$ | ความชันที่ไหลถึง $\mu$ | 23 |
| (35) | $\frac{\partial L_i}{\partial\ell} = -\hat A_t\ \rho_t\ (z^2 - 1), \qquad \frac{\partial H}{\partial\ell} = 1$ | ความชันที่ไหลถึง $\ell$ จาก $L^{CLIP}$ และจาก $H$ | 24 |
| (36) | $\theta \leftarrow \theta - \alpha\,\frac{\hat m_k}{\sqrt{\hat v_k} + \texttt{eps}}$ | Adam ก้าวลูกบิดทุกตัว | 25 |

### สมการที่มีเลขทั้งหมด

เล่มนี้มีสมการที่มีเลข 38 ข้อ คือ (1) ถึง (37) และ (G) เรียงตามเลข ซึ่งไม่ใช่ลำดับที่พบในเล่ม

**(1)** Markov property · บทที่ 6

$$P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0, a_0)$$

**(2)** โจทย์ของ RL · บทที่ 8

$$\pi^* = \arg\max_{\pi}\ \mathbb{E}\Big[\sum_{t=0}^{T} \gamma^t\,r_t\Big]$$

**(3)** นิยามของ value function · บทที่ 9

$$V^{\pi}(s) = \mathbb{E}_{\pi}\Big[\sum_{k \ge 0} \gamma^{k}\,r_{t+k} \ \Big|\ s_t = s\Big]$$

**(4)** สมการ Bellman · บทที่ 9

$$V^{\pi}(s) = \mathbb{E}\big[r_t + \gamma\,V^{\pi}(s_{t+1}) \mid s_t = s\big]$$

**(5)** นิยามของ action-value · บทที่ 9

$$Q^{\pi}(s, a) = \mathbb{E}_{\pi}\Big[\sum_{k \ge 0} \gamma^{k}\,r_{t+k} \ \Big|\ s_t = s,\ a_t = a\Big]$$

**(6)** action-value ในรูปก้าวเดียว · บทที่ 9

$$Q^{\pi}(s, a) = \mathbb{E}\big[r_t + \gamma\,V^{\pi}(s_{t+1}) \mid s_t = s,\ a_t = a\big]$$

**(7)** value function ในรูปของ return · บทที่ 9

$$V^{\pi}(s) = \mathbb{E}\big[G_t \mid s_t = s\big]$$

**(8)** นิยามของ advantage · บทที่ 9

$$A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$$

**(9)** policy ของ Tron1 · บทที่ 11

$$\pi_\theta(\cdot \mid s) = \mathcal{N}\big(\mu_\theta(s),\ \sigma\big), \qquad \sigma = e^{\ell}$$

**(10)** หนึ่ง node · บทที่ 13

$$y = f\big(w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b\big)$$

**(11)** ELU · บทที่ 13

$$\text{ELU}(x) = \begin{cases} x & x > 0 \\ e^{x} - 1 & x \le 0 \end{cases}$$

**(12)** gradient descent · บทที่ 14

$$\theta \leftarrow \theta - \alpha\,\nabla_\theta L$$

**(13)** ความหนาแน่นของ Gaussian · บทที่ 10

$$p(a) = \frac{1}{\sigma\sqrt{2\pi}}\,\exp\!\Big(-\frac{(a-\mu)^2}{2\sigma^2}\Big)$$

**(14)** log ของความหนาแน่นสามพจน์ · บทที่ 10

$$\log p(a) = -\tfrac{1}{2}\ln 2\pi - \ln\sigma - \frac{(a-\mu)^2}{2\sigma^2}$$

**(15)** รวมแปดข้อต่อที่อิสระต่อกัน · บทที่ 11

$$\log\pi_\theta(a \mid s) = \sum_{i=1}^{8} \log\pi(a_i \mid s)$$

**(16)** การสุ่ม action · บทที่ 12

$$a = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 1)$$

**(17)** action เป็นมุมเป้าหมาย · บทที่ 15

$$q_{\text{target}} = 0.25\,a + q_{\text{default}}$$

**(18)** ตัวควบคุม PD · บทที่ 15

$$\tau = k_p\,(q_{\text{target}} - q) - k_d\,\dot q$$

**(19)** TD error · บทที่ 17

$$\delta_t = r_t + (1 - \text{done}_t)\,\gamma\,V(s_{t+1}) - V(s_t)$$

**(20)** importance sampling · บทที่ 18

$$\mathbb{E}_{a \sim p}\big[f(a)\big] = \mathbb{E}_{a \sim q}\Big[\frac{p(a)}{q(a)}\,f(a)\Big]$$

**(21)** probability ratio · บทที่ 18

$$\rho_t = \frac{\pi_{new}(a_t \mid s_t)}{\pi_{old}(a_t \mid s_t)} = \exp\!\big(\log\pi_{new}(a_t \mid s_t) - \log\pi_{old}(a_t \mid s_t)\big)$$

**(22)** เอกลักษณ์ของ score function · บทที่ 19

$$\nabla_\theta\,\pi_\theta(a \mid s) = \pi_\theta(a \mid s)\ \nabla_\theta \log\pi_\theta(a \mid s)$$

**(23)** policy gradient · บทที่ 19

$$\nabla_\theta J(\theta) = \mathbb{E}\Big[\nabla_\theta \log\pi_\theta(a_t \mid s_t)\ A^{\pi}(s_t, a_t)\Big]$$

**(24)** ทิศของ $\mu$ · บทที่ 19

$$\frac{\partial \log\pi_\theta(a \mid s)}{\partial\mu} = \frac{a - \mu}{\sigma^2}$$

**(25)** สมการ GAE แบบผลรวม · บทที่ 22

$$A_t = \sum_{k=0}^{\infty}(\gamma\lambda)^k\,\delta_{t+k}$$

**(26)** สมการ GAE แบบเวียนเกิด · บทที่ 22

$$A_t = \delta_t + (1-\text{done}_t)\,\gamma\lambda\,A_{t+1}, \qquad A_T = 0$$

**(27)** สมการเป้าของ critic · บทที่ 22

$$R_t = A_t + V_t$$

**(28)** สมการ advantage ที่ normalize แล้ว · บทที่ 22

$$\hat A_t = \frac{A_t - \text{mean}(A)}{\text{std}(A) + 10^{-8}}$$

**(29)** clipped surrogate loss · บทที่ 23

$$L^{CLIP} = \text{mean}\Big[\max\big(-\rho_t\,\hat A_t,\ \ -\text{clip}(\rho_t,\ 1-\epsilon,\ 1+\epsilon)\,\hat A_t\big)\Big]$$

**(30)** KL ของระฆัง Gaussian · บทที่ 26

$$D_{KL}\big(\pi_{old} \,\Vert\, \pi_{new}\big) = \sum_{i}\left[\ln\frac{\sigma_{n,i}}{\sigma_{o,i}} + \frac{\sigma_{o,i}^2 + (\mu_{o,i}-\mu_{n,i})^2}{2\,\sigma_{n,i}^2} - \frac12\right]$$

**(31)** entropy ของระฆัง Gaussian · บทที่ 24

$$H = \sum_{i}\Big[\tfrac12 + \tfrac12\ln 2\pi + \ln\sigma_i\Big]$$

**(32)** value loss · บทที่ 24

$$L_V = \text{mean}\Big[\max\big((V_{new} - R_t)^2,\ (V_{clip} - R_t)^2\big)\Big]$$

**(33)** loss รวมของ PPO · บทที่ 24

$$L = L^{CLIP} + c_V\,L_V - c_H\,H$$

**(34)** ระยะมองของ GAE · บทที่ 27

$$1 + \gamma\lambda + (\gamma\lambda)^2 + \cdots = \frac{1}{1-\gamma\lambda}$$

**(35)** ทิศของ $\sigma$ · บทที่ 24

$$\frac{\partial L_i}{\partial\ell} = -\hat A_t\ \rho_t\ (z^2 - 1), \qquad \frac{\partial H}{\partial\ell} = 1$$

**(36)** Adam · บทที่ 25

$$\theta \leftarrow \theta - \alpha\,\frac{\hat m_k}{\sqrt{\hat v_k} + \texttt{eps}}$$

**(37)** กฎปรับ learning rate · บทที่ 26

$$\alpha \leftarrow \begin{cases} \max\!\big(10^{-5},\ \alpha/1.5\big) & \overline{D}_{KL} > 0.02 \\ \min\!\big(10^{-2},\ 1.5\,\alpha\big) & 0 < \overline{D}_{KL} < 0.005 \\ \alpha & 0.005 \le \overline{D}_{KL} \le 0.02 \end{cases}$$

**(G)** ความชันของ loss เทียบกับ $\mu$ · บทที่ 23

$$\frac{\partial L_i}{\partial\mu} = -\hat A_t\ \rho_t\ \frac{a_t - \mu}{\sigma^2}$$
