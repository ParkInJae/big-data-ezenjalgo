<%@ page language="java" contentType="text/html; charset=UTF-8"
         pageEncoding="UTF-8"%>
<%@ include file="head.jsp" %>
<style type="text/css">
.ccard:hover
{
	transform: translate(-7px, -7px);
	transition-duration: 0.3s;
}
</style>
		<div style="height:270px;"></div>
		<table border="0" style="width:85%; background-color:#E8E8E8;">
			<tr>
				<td class="icontent" align="center">
					<div class="ccard">
						<a href="view.jsp">
							<div class="ccard-border-top"></div>
							<div class="img">
								<img src="img/cat2.jpg" style="width:100%; height:100%">
							</div>
							<span> 이젠 그린</span>
							<p class="job">100,000원</p>
						</a>
						<button onClick="getkart()">상품비교 담기</button>
					</div>
				</td>
				<td class="icontent">
					<div class="ccard">
						<a href="view.jsp">
							<div class="ccard-border-top"></div>
							<div class="img">
								<img src="img/cat2.jpg" style="width:100%; height:100%">
							</div>
							<span> 제품 이름</span>
							<p class="job">100,000원</p>
						</a>
						<button onClick="getkart()">상품비교 담기</button>
					</div>
				</td>
				<td class="icontent">
					<div class="ccard">
						<a href="view.jsp">
							<div class="ccard-border-top"></div>
							<div class="img">
								<img src="img/cat2.jpg" style="width:100%; height:100%">
							</div>
							<span> 제품 이름</span>
							<p class="job">100,000원</p>
						</a>
						<button onClick="getkart()">상품비교 담기</button>
					</div>
				</td>
				<%@ include file="card.jsp" %>
			</tr>					
			<tr>
				<td class="icontent">
					<div class="ccard">
						<a href="view.jsp">
							<div class="ccard-border-top"></div>
							<div class="img">
								<img src="img/cat2.jpg" style="width:100%; height:100%">
							</div>
							<span> 제품 이름</span>
							<p class="job">100,000원</p>
						</a>
						<button onClick="getkart()">상품비교 담기</button>
					</div>
				</td>
				<td class="icontent">
					<div class="ccard">
						<a href="view.jsp">
							<div class="ccard-border-top"></div>
							<div class="img">
								<img src="img/cat2.jpg" style="width:100%; height:100%">
							</div>
							<span> 제품 이름</span>
							<p class="job">100,000원</p>
						</a>
						<button onClick="getkart()">상품비교 담기</button>
					</div>
				</td>
				<td class="icontent">
					<div class="ccard">
						<a href="view.jsp">
							<div class="ccard-border-top"></div>
							<div class="img">
								<img src="img/cat2.jpg" style="width:100%; height:100%">
							</div>
							<span> 제품 이름</span>
							<p class="job">100,000원</p>
						</a>
						<button onClick="getkart()">상품비교 담기</button>
					</div>
				</td>
			</tr>
			<tr>
				<td class="icontent">
					<div class="ccard">
						<a href="view.jsp">
							<div class="ccard-border-top"></div>
							<div class="img">
								<img src="img/cat2.jpg" style="width:100%; height:100%">
							</div>
							<span> 제품 이름</span>
							<p class="job">100,000원</p>
						</a>
						<button onClick="getkart()">상품비교 담기</button>
					</div>
				</td>
			</tr>									
		</table>
	</body>
</html>