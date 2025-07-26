from rouge import Rouge
import numpy as np
r = Rouge()


def Summarization_Evaluation(abstractve_summarization, reference_text):
  out = r.get_scores(abstractve_summarization, reference_text)
  EVAL_=[]
  for i in range(3): # Rouge-1, Rouge-2, Rouge-3
    Eval = np.zeros((3))
    if i == 2:
      Eval[0] = out[0]['rouge-l']['f']  # F1Score
      Eval[1] = out[0]['rouge-l']['p']  # Precision
      Eval[2] = out[0]['rouge-l']['r']  # Recall
    else:
      Eval[0] = out[0]['rouge-'+str(i+1)]['f'] # F1Score
      Eval[1] = out[0]['rouge-'+str(i+1)]['p'] # Precision
      Eval[2] = out[0]['rouge-'+str(i+1)]['r'] # Recall
    EVAL_.append(Eval)
  return EVAL_


