# Spaceship-Titanic

# gitのやり方
https://qiita.com/non0311/items/369eecea3456dbcbe26e
## 注意点
### add,commit,pushの流れ
git statusでaddするファイルを確認。

すべてaddするなら、git add .

特定のファイルのみをaddするなら、git add <ファイル名>

git commit -m 'コメント'

git push origin <ブランチ名>

その後github上で誰かがmergeする

### 誰かがpushした場合
自分のブランチでadd,commitを完了したうえでローカルのmainブランチで
git pull origin <自分のいるブランチ名>
その後
git merge main

# パッケージの管理
パッケージはpoetryを用いて管理。具体的な管理方法は以下を参照。

## パッケージのインストール
```
poetry install
```
## パッケージを追加したい場合
```
poetry add 'パッケージ名'
```
 ## 仮想環境への出入り
 入り方
 ```
poetry shell
```
出方
```
exit
```

# 実行方法
仮想環境内で
```
python main.py
```


