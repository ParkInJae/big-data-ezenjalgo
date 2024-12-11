<%@ page language="java" contentType="text/html; charset=UTF-8"
         pageEncoding="UTF-8"%>
<%@ include file="head.jsp" %>
<style type="text/css">
/* 제품 목록 */
.ccard {
  width: 90%;
  height: 90%;
  background-image: linear-gradient(144deg,#8608b4, #492fed 60%,#bd6fda);
  border: none;
  border-radius: 10px;
  padding-top: 10px;
  margin: auto;
  font-family: inherit;
}
.ccard .img {
  width: 370px;
  height: 370px;
  background: #e8e8e8;
  border-radius: 100%;
  margin: auto;
  margin-top: 20px;
}
</style> 
		<div style="height:270px;"></div>
		<table border="0" style="width:100%; height:900px; background-color:#E8E8E8;">
			<tr>
				<td>
					<div class="ccard">
						<div class="ccard-border-top"></div>
						<div class="img">
							<img src="img/cat2.jpg" style="width:100%; height:100%">
						</div>
						<span>이젠 사료</span>
						<p class="job">가격 : 100,000원</p>
						<p class="job">별점 : 5.0점</p>
						<p class="job" align="center"> 
							<table border="0" style="width:100%;">
								<tr>
									<th width="200px"></th>
									<td style="color: white">긍정 키워드</td>
									<td style="color: white; width:150px">횟수</td>
									<td rowspan="6">
										<img src="img/wordcloude.png" style="width:300px; height:300px">
									</td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">조아영</td>
									<td style="color: white">44</td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">맛있어용</td>
									<td style="color: white">32</td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">빨라용</td>
									<td style="color: white">30</td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">건강해용</td>
									<td style="color: white">28</td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">평점이</td>
									<td style="color: white">20</td>
								</tr>
							</table>
						</p>
						<p class="job">
							<table border="0" style="width:100%;">
								<tr>
									<th width="200px"></th>
									<td style="color: white">부정 키워드</td>
									<td style="color: white; width:150px">횟수</td>
									<td rowspan="6">
										<img src="img/wordcloude.png" style="width:300px; height:300px">
									</td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">아니</td>
									<td style="color: white">44</td>
									<td></td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">진짜</td>
									<td style="color: white">32</td>
									<td style="color: white"></td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">이게</td>
									<td style="color: white">30</td>
									<td></td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">뭔데</td>
									<td style="color: white">28</td>
									<td></td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">느려</td>
									<td style="color: white">20</td>
									<td></td>
								</tr>
							</table>
						</p>
						<br>
						<br>
						<button onClick="getbuy();">상품 바로 구매하기</button>
						<br>
						<br>
						<br>
					</div>
				</td>
				<td>
					<div class="ccard">
						<div class="ccard-border-top"></div>
						<div class="img">
							<img src="img/cat2.jpg" style="width:100%; height:100%">
						</div>
						<span>이젠 사료</span>
						<p class="job">가격 : 100,000원</p>
						<p class="job">별점 : 5.0점</p>
						<p class="job" align="center"> 
							<table border="0" style="width:100%;">
								<tr>
									<th width="200px"></th>
									<td style="color: white">긍정 키워드</td>
									<td style="color: white; width:150px">횟수</td>
									<td rowspan="6">
										<img src="img/wordcloude.png" style="width:300px; height:300px">
									</td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">조아영</td>
									<td style="color: white">44</td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">맛있어용</td>
									<td style="color: white">32</td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">빨라용</td>
									<td style="color: white">30</td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">건강해용</td>
									<td style="color: white">28</td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">평점이</td>
									<td style="color: white">20</td>
								</tr>
							</table>
						</p>
						<p class="job">
							<table border="0" style="width:100%;">
								<tr>
									<th width="200px"></th>
									<td style="color: white">부정 키워드</td>
									<td style="color: white; width:150px">횟수</td>
									<td rowspan="6">
										<img src="img/wordcloude.png" style="width:300px; height:300px">
									</td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">아니</td>
									<td style="color: white">44</td>
									<td></td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">진짜</td>
									<td style="color: white">32</td>
									<td style="color: white"></td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">이게</td>
									<td style="color: white">30</td>
									<td></td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">뭔데</td>
									<td style="color: white">28</td>
									<td></td>
								</tr>
								<tr>
									<td></td>
									<td style="color: white">느려</td>
									<td style="color: white">20</td>
									<td></td>
								</tr>
							</table>
						</p>
						<br>
						<br>
						<button onClick="getbuy();">상품 바로 구매하기</button>
						<br>
						<br>
						<br>
					</div>
				</td>
			</tr>										
		</table>
	</body>
</html>