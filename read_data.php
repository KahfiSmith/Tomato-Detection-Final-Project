<?php
    $konek = mysqli_connect("localhost", "root", "", "tomatify");
    $sql2 = mysqli_query($konek, "select kematangan from kematangan_tomat ORDER BY id_kematangan DESC LIMIT 1");

      if(mysqli_num_rows($sql2)>0)
      {
        $row = mysqli_fetch_assoc($sql2);
        
        echo $row['kematangan'];
      }
?>