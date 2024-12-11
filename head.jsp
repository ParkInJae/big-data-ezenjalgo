<%@ page language="java" contentType="text/html; charset=UTF-8"
         pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
	<head>
		<meta charset="UTF-8">
		<title>EZEN GALGO</title>
		<link rel="stylesheet" href="CSS.css">
		<script src="https://cdn.jsdelivr.net/npm/sweetalert2@9"></script>
	</head>
<style type="text/css">

</style>
<script>
function biguyo()
{
	location.href="biguyo.jsp"
}

function getkart()
{	
	 // .card__content 요소를 찾습니다.
    const cardContent = document.querySelector('.card__content');

    // .card__content 요소 내의 div 요소를 찾습니다.
    const divs = cardContent.querySelectorAll('div');
    
    if (divs.length >= 4)
    {
        /* alert('상품 비교칸에는 3개이상 담을수 없습니다. \n상품을 제거하시려면 해당상품을 클릭해주세요.'); */
    	Swal.fire("상품 비교칸에는 3개이상 담을수 없습니다. \n상품을 제거하시려면 해당상품을 클릭해주세요.");
    }else
	{
		// 새로운 div 요소를 생성합니다.
		const newDiv1 = document.createElement('div');
		const newDiv2 = document.createElement('div');
	
		// 새로운 img 요소를 생성합니다.
		const newImg = document.createElement('img');
		const newText = document.createTextNode('이젠 사료');
		
		// img 요소의 속성을 설정합니다.
		newImg.src = 'img/cat2.jpg';
		newText.text = ('이젠 사료');
		
		newImg.style.width = '200px';
	    newImg.style.height = '200px';
	    
		// img 요소를 div에 추가합니다.
		newDiv1.appendChild(newImg);
		newDiv2.appendChild(newText);
		
		// 클릭 이벤트 리스너를 div에 추가하여, 클릭 시 div를 삭제합니다.
		newDiv1.addEventListener('click', function()
		{
		    newDiv1.remove();
		    newDiv2.remove();
		});
		const cardContent = document.querySelector('.card__content');
	
		cardContent.appendChild(newDiv1);
		cardContent.appendChild(newDiv2);
	}
}
</script> 
	<body>
		<table class="main_banner">
			<tr>
				<td width="250px">
					<a href="index.jsp">
						<img src="img/all.png" style="width:65%; height:65%;">
					</a>
				</td>
				<td width="250px;">
					<a href="index.jsp">
						<img src="img/catsrang.jpg" style="width:65%; height:65%;">
					</a>
				</td>
				<td width="250px">
					<a href="index.jsp">
						<img src="img/nutrena.jpg" style="width:65%; height:65%;">
					</a>
				</td>
				<td width="250px">
					<a href="index.jsp">
						<img src="img/royalcanin.png" style="width:65%; height:65%;">
					</a>
				</td>
				<td width="250px">
					<a href="index.jsp">
						<img src="img/purevita.jpg" style="width:65%; height:65%;">
					</a>
				</td>
				<td width="250px">
					<a href="index.jsp">
						<img src="img/naturalbalance.png" style="width:65%; height:65%;">
					</a>
				</td>
			</tr>
		</table>