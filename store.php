<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
</html>
<?php
#$connect=mysqli_connect('localhost','id9317733_kjaved8085','javed8085','id9317733_kjaved8085');
$folder="images/in/";

        
$pic1=$_FILES["image22"]["name"];
$tmppic=$_FILES["image22"]["tmp_name"];


if(move_uploaded_file($tmppic,$folder.$pic1)){
    
echo "image uploaded";
#echo "<script>location.href='phppy.py';</script>";
echo "<script>location.href='python1.py';</script>";
}
?>




