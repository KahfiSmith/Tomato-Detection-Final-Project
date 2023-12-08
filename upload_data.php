<?php

var_dump($_POST);

// $target_dir = "captured_images/"; // folder untuk menyimpan gambar
// $target_filename = "Photo.jpg"; // set the constant filename without an extension
// $target_file = $target_dir . $target_filename;
// $uploadOk = 1;
// $imageFileType = strtolower(pathinfo($_FILES["imageFile"]["name"], PATHINFO_EXTENSION));
// $file_name = pathinfo($_FILES["imageFile"]["name"], PATHINFO_BASENAME);

$target_dir = "captured_images/";
$date = new DateTime();
$date_string = $date->format('H-i-s Y-m-d');
$target_file = $target_dir . $date_string . basename($_FILES["imageFile"]["name"]);
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file, PATHINFO_EXTENSION));
$file_name = pathinfo($target_file, PATHINFO_BASENAME);

// Debugging: Print entire POST data
var_dump($_POST);

// ... (your existing code)

if ($uploadOk == 0) {
    echo "Sorry, your file was not uploaded.";
} else {
    if (move_uploaded_file($_FILES["imageFile"]["tmp_name"], $target_file)) {
        echo "Photo berhasil dipuload di server dengan nama " . $file_name;

        $servername = "localhost";
        $username = "root";
        $password = "";
        $dbname = "tomatify";

        $conn = new mysqli($servername, $username, $password, $dbname);
        if ($conn->connect_error) {
            die("Connection failed: " . $conn->connect_error);
        }

        $berat = isset($_POST["berat"]) ? mysqli_real_escape_string($conn, $_POST["berat"]) : "";
        $sql = "INSERT INTO berat_tomat (berat) VALUES ('$berat')";
        echo "SQL Query: " . $sql;

        $stmt = $conn->prepare("INSERT INTO berat_tomat (berat) VALUES (?)");
        $stmt->bind_param("s", $berat);

        if ($stmt->execute()) {
            echo "Data berhasil dimasukkan ke database. Berat: " . $berat ;
        } else {
            echo "Error: " . $stmt->error;
        }

        $stmt->close();
        $conn->close();
    } else {
        echo "Sorry, Ada error dalam proses upload photo.";
    }
}
?>