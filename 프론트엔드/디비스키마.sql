create table tb_board( id bigint auto_increment primary key, 
                       title varchar(300) not null,
                       contents longtext, 
                       filename varchar(300),
                       image_url varchar(300), 
                       writer varchar(100),
                       wdate datetime, 
                       hit int );
                       
mysql -u root -p 
Enter your password : 1234
use mydb;


insert into tb_board(title, contents, writer, wdate, hit)
values('제목1', '내용1', 'test1', now(), 0);
insert into tb_board(title, contents, writer, wdate, hit)
values('제목2', '내용2', 'test1', now(), 0);
insert into tb_board(title, contents, writer, wdate, hit)
values('제목3', '내용3', 'test1', now(), 0);
insert into tb_board(title, contents, writer, wdate, hit)
values('제목4', '내용4', 'test1', now(), 0);
insert into tb_board(title, contents, writer, wdate, hit)
values('제목5', '내용5', 'test1', now(), 0);

cmd - 관리자권한
conda activate backend 
conda install pymysql sqlalchemy 