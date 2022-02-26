# Text_Image_Recommendation

입력된 텍스트의 감정과 어울리는 이미지를 추천해주는 추천기입니다.


<img width="753" alt="스크린샷 2022-02-26 오후 7 25 23" src="https://user-images.githubusercontent.com/62793657/155839697-b9fac8a8-b141-4ad2-9458-96f1db65e659.png">

## 1. 입력된 텍스트의 감정분석 - Text_Sentimental_Analysis
- DataSet : train / test
- AI 허브의 단발성 대화데이터셋과 네이버 뉴스/유튜브 댓글을 크롤링하여 데이터셋을 구성하였습니다.
- 모델은 KcBert를 이용하였고, 입력된 텍스트를 슬픔/기쁨/놀람/두려움/역겨움/분노 6가지 감정으로 구분합니다.

@inproceedings{lee2020kcbert,
  title={KcBERT: Korean Comments BERT},
  author={Lee, Junbum},
  booktitle={Proceedings of the 32nd Annual Conference on Human and Cognitive Language Technology},
  pages={437--440},
  year={2020}
}
