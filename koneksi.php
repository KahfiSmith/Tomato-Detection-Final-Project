<?php
class berat_tomat{
 public $link='';
 function __construct($berat){
  $this->connect();
  $this->storeInDB($berat);
 }
 
 function connect(){
  $this->link = mysqli_connect('localhost','root','') or die('Cannot connect to the DB');
  mysqli_select_db($this->link,'tomatify') or die('Cannot select the DB');
 }
 
 function storeInDB($berat){
  $query = "insert into berat_tomat set berat='".$berat."' ";
  $result = mysqli_query($this->link,$query) or die('Errant query:  '.$query);
 }
 
}
if($_GET['berat'] != ''){
 $berat_tomat=new berat_tomat($_GET['berat']);
}


?>