var input;
var names = [];

function insertName()
{
	input = document.getElementById("userInput").value;
	console.log(input)
	names.push(input)

	console.log(names)
	clearAndShow();
}

// how to check for which name needs to be removed
function removeName(i){
	// remove the li
	$('#'+ i).remove();

	// remove name from names list
	names.splice(i)
	console.log(names)
}

function clearAndShow () {
	document.getElementById("show").innerHTML = "Ingevulde namen:";
	value = input;

	// get index of name in list
	console.log(names)

	let nameIndex = names.indexOf(value)
	$('#list').append('<li id="'+nameIndex+'">'+value+'<button class="button button3" onclick="removeName('+nameIndex+')"><i class="fa fa-close"></i></button></li>'); 
}

function getAllWords(sentence) {
  return typeof sentence === 'string' && sentence.length > 0 ?
         sentence.split(' ') : [];
}

function bouwRooster() {
	if (!rooster){
		alert('Upload rooster aub')
	}
	else if (!input){
		alert('Vul je naam in aub')
	}
	else if (names.length == 0){
		alert('Vul je naam in aub')
	}
	else {

		// initialize csv headers
		let csvContent = "data:text/csv;charset=utf-8\n";
		let header = ",subject,start date,start time,end date,end time,location"
		csvContent += header + "\r\n";
		let location = "Koningsstraat 54"

		let year = new Date().getFullYear()
		let counter = 0;

		var maanden = ['jan', 'feb', 'maart', 'april', 'mei', 'juni', 'juli', 'aug', 'sep', 'okt' , 'nov', 'dec']
		
		rooster.forEach(function(row, i){
			// check of er data te vinden zijn

			row.forEach(function(dag, j){
				// vindt de data
				let month;
				let day;

				foo = getAllWords(dag)
				if (foo.length == 2){
					day = foo[0].replace(/\s+/g, '');
					month = foo[1].toLowerCase().replace(/\s+/g, '');					
				}
				if (maanden.includes(month)){

					var monthIndex = maanden.indexOf(month) + 1
					for (k = 1; k < 5; k++) {
						//console.log(rooster[i+k][j].toLowerCase())

						// loop over names
						for (h=0; h < names.length; h++){

						 	if (rooster[i+k][j].toLowerCase().replace(/\s+/g, '') == names[h].toLowerCase()){
						 		counter += 1;

						 		let csvRow = "";
						 		let subject;
						 		let startDate;
						 		let startTime;
						 		let endDate
						 		let endTime;
								// bar
								if (k==1){
									subject = "Werken van Beeren (bar)"
									startDate = day + '/' + String(monthIndex) + '/' + String(year)
									startTime = "17:00"
									endDate = day + '/' + String(monthIndex) + '/' + String(year)
									endTime = "23:59"
								}
								// 1ste
								else if (k==2){
									subject = "Werken van Beeren (1ste)"
									startDate = day + '/' + String(monthIndex) + '/' + String(year)
									startTime = "15:30"
									endDate = day + '/' + String(monthIndex) + '/' + String(year)
									endTime = "22:00"
								}
								// 2de
								else if (k==3){
									subject = "Werken van Beeren (2de)"
									startDate = day + '/' + String(monthIndex) + '/' + String(year)
									startTime = "18:00"
									endDate = day + '/' + String(monthIndex) + '/' + String(year)
									endTime = "23:00"
								}
								else {
									subject = "Werken van Beeren (3de)"
									startDate = day + '/' + String(monthIndex) + '/' + String(year)
									startTime = "19:00"
									endDate = day + '/' + String(monthIndex) + '/' + String(year)
									endTime = "23:30"
								}

								// combine row
								let newRow = [subject, startDate, startTime, endDate, endTime, location]

							    let fullNewRow = newRow.join(",");
							    csvContent += fullNewRow + "\r\n";
							}
					 	}
					}

				}
			});
		});

		if (counter == 0){
			alert('Niets gevonden in het rooster!')
		}
		else {
			var encodedUri = encodeURI(csvContent);
			var link = document.createElement("a");
			link.setAttribute("href", encodedUri);
			link.setAttribute("download", "mijn_rooster.csv");
			document.body.appendChild(link); // Required for FF

			link.click(); // This will download the data file named "my_data.csv"
		}
	}

}

