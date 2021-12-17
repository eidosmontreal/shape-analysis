DATADIR=$1
mkdir -p $DATADIR

# Download and unzip data
wget http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Tele-aliens/shapes.zip && wget http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Tele-aliens/gt.zip
mkdir -p $DATADIR/tele_aliens
unzip shapes.zip -d $DATADIR/tele_aliens && unzip gt.zip -d $DATADIR/tele_aliens
rm shapes.zip && rm gt.zip

wget http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Vases/shapes.zip && wget http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Vases/gt.zip
mkdir -p $DATADIR/vases
unzip shapes.zip -d $DATADIR/vases && unzip gt.zip -d $DATADIR/vases
rm shapes.zip && rm gt.zip

wget http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Chairs/shapes.zip && wget http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Chairs/gt.zip
mkdir -p $DATADIR/chairs
unzip shapes.zip -d $DATADIR/chairs && unzip gt.zip -d $DATADIR/chairs
rm shapes.zip && rm gt.zip


